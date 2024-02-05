/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by original author
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

\*---------------------------------------------------------------------------*/

#include "linGeomTotalDispSolidML.H"
#include "fvm.H"
#include "fvc.H"
#include "fvMatrices.H"
#include "addToRunTimeSelectionTable.H"
#include "linearElastic.H"
#include "momentumStabilisation.H"
#define FDEEP_FLOAT_TYPE double
#include <fdeep/fdeep.hpp>
#include "fvCFD.H" 
#include <Eigen/Dense>
#include <cmath>


// #include "FoamFile.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    namespace solidModels
    {

        // * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

        defineTypeNameAndDebug(linGeomTotalDispSolidML, 0);
        addToRunTimeSelectionTable(solidModel, linGeomTotalDispSolidML, dictionary);

        // * * * * * * * * * * *  Private Member Functions * * * * * * * * * * * * * //

        void linGeomTotalDispSolidML::predict()
        {
            Info << "Linear predictor using DD" << endl;

            // Predict D using the increment of displacement field from the previous
            // time-step
            D() = D().oldTime() + U() * runTime().deltaT();

            // Update gradient of displacement
            mechanical().grad(D(), gradD());

            // Calculate the stress using run-time selectable mechanical law
            mechanical().correct(sigma());
        }

        // * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

        linGeomTotalDispSolidML::linGeomTotalDispSolidML(
            Time &runTime,
            const word &region)
            : solidModel(typeName, runTime, region),
              impK_(mechanical().impK()),
              impKf_(mechanical().impKf()),
              rImpK_(1.0 / impK_),
              predictor_(solidModelDict().lookupOrDefault<Switch>("predictor", Switch(false))),
              indivResidualFilePtr_(),
              writeIndivResidualFile_(
                  solidModelDict().lookupOrDefault<Switch>("writeIndivResidualFile", Switch(false))),
              machineLearning_(
                  solidModelDict().lookupOrDefault<Switch>("machineLearning", Switch(false))),
              useResiduals_(
                  solidModelDict().lookupOrDefault<Switch>("useResiduals", Switch(false))),
              machinePredictorIter_(
                  solidModelDict().lookupOrDefault<scalar>("iterationToApplyMachineLearningPredictor", 20)),
              jsonFiles_(
                  solidModelDict().lookupOrDefault<List<fileName>>("jsonFiles", List<fileName>(1, "jsonFile"))),
            inputsML_(
                  solidModelDict().lookupOrDefault<List<fileName>>("inputsML", List<fileName>(1, "D"))),
            convergedCase_(
                  solidModelDict().lookupOrDefault<fileName>("convergedCase", "../plateHoleTotalDisp")),
              prevCellD_(),
              testConverged_(
                  solidModelDict().lookupOrDefault<Switch>("testConverged", Switch(false))),
              BCloopCorrFile_(),
                errorDcellsFile_(),
              DcellsFile_(),
              writeCellDisplacement_(
                  solidModelDict().lookupOrDefault<Switch>("writeCellDisplacement", Switch(false))),
              writeFields_(
                  solidModelDict().lookupOrDefault<Switch>("writeFields", Switch(true))),
              cellDisplacementFile_(),
              writeDisplacementField_(
                  solidModelDict().lookupOrDefault<Switch>("writeDisplacementField", Switch(false))),
              writeDisplacementLimit_(
                  solidModelDict().lookupOrDefault<scalar>("writeDisplacementLimit", 20)),
              useCoordinates_(
                  solidModelDict().lookupOrDefault<Switch>("useCoordinates", false)),
              predictZ_(
                  solidModelDict().lookupOrDefault<Switch>("predictZ", true)),
              noPredictions_(
                  solidModelDict().lookupOrDefault<scalar>("noPredictions", 1)),
                relaxD_ML_(solidModelDict().lookupOrDefault<scalar>("relaxD_ML", 1.0)),
              debugFields_(
                  solidModelDict().lookupOrDefault<Switch>("debugFields", false)),
              tractionBCtol_(
                  solidModelDict().lookupOrDefault<scalar>("tractionBCtolerance", 1e-10)),
                muf_
                    (
                        IOobject
                        (
                            "interpolate(mu)",
                            runTime.timeName(),
                            mesh(),
                            IOobject::NO_READ,
                            IOobject::NO_WRITE
                        ),
                        mesh(),
                        dimensionedScalar("0", dimPressure, 0.0)
                    ),
                    lambdaf_
                    (
                        IOobject
                        (
                            "interpolate(lambda)",
                            runTime.timeName(),
                            mesh(),
                            IOobject::NO_READ,
                            IOobject::NO_WRITE
                        ),
                        mesh(),
                        dimensionedScalar("0", dimPressure, 0.0)
                    )
        {
            DisRequired();

            // We will directly read the linearElastic mechanicalLaw
            const PtrList<mechanicalLaw>& mechLaws = mechanical();
            // if (mechLaws.size() != 1)
            // {
            //     FatalErrorIn
            //     (
            //         "coupledUnsLinGeomLinearElasticSolid::"
            //         "coupledUnsLinGeomLinearElasticSolid"
            //     )   << type() << " can currently only be used with a single material"
            //         << "\nConsider using one of the other solidModels."
            //         << abort(FatalError);
            // }
            // else if (!isA<linearElastic>(mechLaws[0]))
            // {
            //     FatalErrorIn
            //     (
            //         "coupledUnsLinGeomLinearElasticSolid::"
            //         "coupledUnsLinGeomLinearElasticSolid"
            //     )   << type() << " can only be used with the linearElastic "
            //         << "mechanicalLaw" << nl
            //         << "Consider using one of the other linearGeometry solidModels."
            //         << abort(FatalError);
            // }

            // Cast the mechanical law to a linearElastic mechanicalLaw
            const linearElastic& mech = refCast<const linearElastic>(mechLaws[0]);

            // // Set mu and lambda fields
            muf_ = mech.mu();
            lambdaf_ = mech.lambda();

            // Force all required old-time fields to be created
            fvm::d2dt2(D());

            // For consistent restarts, we will calculate the gradient field
            D().correctBoundaryConditions();
            D().storePrevIter();
            mechanical().grad(D(), gradD());

            if (Pstream::master() && writeIndivResidualFile_)
            {
                Info << "Creating residualTime.dat" << endl;
                indivResidualFilePtr_.set(
                    new OFstream(runTime.path() / "residualIndividual.dat"));
                indivResidualFilePtr_()
                    << "Iter Time Res_x Res_y Res_z" << endl;
            }

            if (predictor_)
            {
                // Check ddt scheme for D is not steadyState
                const word ddtDScheme(
#ifdef OPENFOAMESIORFOUNDATION
                    mesh().ddtScheme("ddt(" + D().name() + ')')
#else
                    mesh().schemesDict().ddtScheme("ddt(" + D().name() + ')')
#endif
                );

                if (ddtDScheme == "steadyState")
                {
                    FatalErrorIn(type() + "::" + type())
                        << "If predictor is turned on, then the ddt(" << D().name()
                        << ") scheme should not be 'steadyState'!" << abort(FatalError);
                }
            }

            predictionResiduals_ = List<scalar>(solidModelDict().lookup("predictionResiduals"));

            inputSizes = List<scalar>(inputsML_.size(), 0.0);

            // Store input feature fields (both ML and base)
            initMLFieldStoring();
      
            if (machineLearning_)
            {

                // Load Scaling values
                kerasScalingMeans_ = List<scalarField>(
                    solidModelDict().lookup("kerasScalingMeans"));

                kerasScalingStds_ = List<scalarField>(
                    solidModelDict().lookup("kerasScalingStds"));        


                // prevCellD_.setSize(machinePredictorIter_);
                // forAll(prevCellD_, iterI)
                // {
                //     prevCellD_.set(
                //         iterI,
                //         new vectorField(D().internalField().size(), vector::zero) // hard coding workaround
                //     );
                // }

                // if (useResiduals_)
                // {
                //     prevCellR_.setSize(machinePredictorIter_);
                //     forAll(prevCellR_, iterI)
                //     {
                //         prevCellR_.set(
                //             iterI,
                //             new vectorField(D().internalField().size(), vector::zero)
                //         );
                //     }
                // }
            }

            debug_cells.append(23629);
            debug_cells.append(118255);
            debug_cells.append(307172);
            debug_cells.append(497682);
            debug_cells.append(599029);
            debug_cells.append(623630);

            // Initiate BC Correction Loop File
            if (Pstream::master())
            {
                Info << "Creating BCloopCorr.dat" << endl;
                BCloopCorrFile_.set(
                    new OFstream(runTime.path() / "BCloopCorr.dat"));

                BCloopCorrFile_()
                    << "Time" << tab
                    << "Converged residual " << tab
                    << "BCloopCorr" << endl;

                if (debugFields_)
                {
                    errorDcellsFile_.set(
                        new OFstream(runTime.path() / "errorDcellsFile.dat"));
                    DcellsFile_.set(
                        new OFstream(runTime.path() / "DcellsFile.dat"));
        
                    errorDcellsFile_() << "Iter" << tab;
                    DcellsFile_() << "Iter" << tab;
        
                    forAll(debug_cells, cellI)
                    {
                        errorDcellsFile_() << debug_cells[cellI] << "x" << tab;
                        errorDcellsFile_() << debug_cells[cellI] << "y" << tab;
                        errorDcellsFile_() << debug_cells[cellI] << "z" << tab;

                        DcellsFile_() << debug_cells[cellI] << "x" << tab;
                        DcellsFile_() << debug_cells[cellI] << "y" << tab;
                        DcellsFile_() << debug_cells[cellI] << "z" << tab;
                    }
                    errorDcellsFile_() << endl;
                    DcellsFile_() << endl;
                }
            }

            // Initiate Cell List File
            if (writeCellDisplacement_)
            {
                cellList_ = List<scalar>(
                    solidModelDict().lookup("cellList"));

                if (Pstream::master())
                {
                    Info << "Creating writeCellDisplacement.dat" << endl;
                    cellDisplacementFile_.set(
                        new OFstream(runTime.path() / "CellDisplacement.dat"));

                    forAll(cellList_, cell)
                    {
                        cellDisplacementFile_() << cellList_[cell] << tab << "X" << tab << "Y" << tab << "Z" << tab;
                    };

                    cellDisplacementFile_() << endl;
                }
            }

            // OFstream BCloopCorrFile("BCloopCorr.dat");
            // BCloopCorrFile << "Time" << tab << "Converged residual "<< tab <<  "BCloopCorr" << endl;
    
        }

        // * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

        bool linGeomTotalDispSolidML::evolve()
        {
            int startIter = 0;
            int endIter;
            predictionCount = 0;

            // Make prediction is switch is on
            bool predictionSwitch = false;

            Info << "Evolving solid solver" << endl;

            if (predictor_)
            {
                predict();
            }

            // Mesh update loop
            do
            {
                iCorr = 0;
#ifdef OPENFOAMESIORFOUNDATION
                SolverPerformance<vector> solverPerfD;
                SolverPerformance<vector>::debug = 0;
#else
                lduSolverPerformance solverPerfD;
                blockLduMatrix::debug = 0;
#endif

                if (useCoordinates_)
                {
                    // Write cell centre coordinates

                    vectorIOField dataToWrite(
                        IOobject(
                            "cellCentres",
                            runTime().timeName(),
                            runTime(),
                            IOobject::NO_READ,
                            IOobject::AUTO_WRITE),
                        mesh().C());

                    dataToWrite.write();
                }

                Info << "Solving the momentum equation for D" << endl;

                scalarField prevResiduals;
                prevResiduals = scalarField(5, 0);
                scalarField prevSlopes;
                prevSlopes = scalarField(5, 0);
                scalarField slopePrecentage;
                slopePrecentage = scalarField(4, 0);
                
                bool smoothConvergence; 
                smoothConvergence = true;
                int smoothConvergenceCount = 0;

                // Momentum equation loop
                do
                {

                    // scalar residual_print = residualvf();

                    // forAll(prevResiduals, cellI)
                    // {
                    //     if (cellI < 4)
                    //     {
                    //         prevResiduals[cellI]=prevResiduals[cellI + 1];
                    //         prevSlopes[cellI]=prevSlopes[cellI + 1];
                    //     }
                    //     else
                    //     {
                    //         prevResiduals[cellI]=residual_print;                          
                    //     }                        
                    // }        

                    // Eigen::VectorXd x(5);
                    // x << 0, 1, 2, 3, 4;
                   
                    // Eigen::VectorXd y(5);
                    // // Fill y with the corresponding values based on your logic
                    // // For example:
                    // y << prevResiduals[0], prevResiduals[1], prevResiduals[2], prevResiduals[3], prevResiduals[4];

                    // if (min(prevResiduals) > 0)
                    // {

                    //     // Take the logarithm with base 10 of y-values
                    //     for (int i = 0; i < y.size(); i++) {
                    //         y(i) = std::log10(y(i));
                    //     }

                    //     // Perform linear regression to find the slope
                    //     Eigen::VectorXd ones = Eigen::VectorXd::Ones(5);
                    //     Eigen::MatrixXd A(5, 2);
                    //     A << x, ones;
                    //     Eigen::VectorXd slope = (A.transpose() * A).ldlt().solve(A.transpose() * y);


                    //     // The first element of 'slope' is the slope of the line
                    //     double line_slope = slope(0);
                    //     // Info << "Slope of the line through the first 5 elements in prevResiduals: " << line_slope << endl;
                    //     prevSlopes[4] = line_slope;
                        

                    //     forAll(slopePrecentage, cell)
                    //     {
                    //         slopePrecentage[cell] = sqrt(pow(prevSlopes[cell]-prevSlopes[4], 2))/sqrt(pow(prevSlopes[4],2));
                    //     }

                    //     // Info << "prevSlopes: " << prevSlopes << endl;
                    //     // Info << "slopePrecentage: " << slopePrecentage << endl;
                    //     // Info << "max(slopePrecentage): " << max(slopePrecentage) << endl;

                    //     if(max(slopePrecentage) < 0.5)
                    //     {
                    //         smoothConvergence = true;
                    //         // Info << "Slopes are smooth: " << "max(slopePrecentage): " << max(slopePrecentage)  << endl;
                    //     }
                    //     else
                    //     {
                    //         smoothConvergence = false;
                    //         // Info << "Slopes are NOT smooth: " << "max(slopePrecentage): " << max(slopePrecentage)  << endl;
                    //     }
                    // }

                    scalar residual_print = residualvf();

                    prevResiduals[0]=prevResiduals[1];
                    prevResiduals[1]=prevResiduals[2];
                    prevResiduals[2]=residual_print;            

                    scalar diff01 =  prevResiduals[0] - prevResiduals[1];
                    scalar diff12 =  prevResiduals[1] - prevResiduals[2];

                    scalar diff01percentage = sqrt(pow( (diff01/(prevResiduals[0]+SMALL)) , 2));
                    scalar diff12percentage = sqrt(pow( (diff12/(prevResiduals[1]+SMALL)) , 2));


                    // Check if the last two residual changes have been smaller than 10 percent
                    scalar threshold = 0.1;
                    if ((diff01percentage < threshold) && (diff12percentage < threshold))
                    {
                        smoothConvergence = true;
                        smoothConvergenceCount = smoothConvergenceCount + 1;

                    }
                    else
                    {
                        smoothConvergence = false;
                        smoothConvergenceCount = 0;
                    }

                    // Info << "diff01percentage: "  << diff01percentage << " ; diff12percentage:" << diff12percentage << " ; smoothConvergenceCount: " << smoothConvergenceCount << " ; smoothConvergence: " << smoothConvergence <<  endl;
            
                    // The first residual when iCorr=0 is 0
                    // Using this method will never use the resiudal when iCorr=0
                    // This is why the results will be different to linGeomTotatlDispSolidML/linGeomTotatlDispSolidML.C
                    if (predictionCount < noPredictions_ )
                    {
                        if ((residualvf() < predictionResiduals_[predictionCount]) && (iCorr > 0))
                        {
                            if (smoothConvergence | predictionCount==0 )
                            {
                                Info << "At iteration: " << iCorr << " residual: " << residual_print
                                    << " is less that prediction residual: " << predictionResiduals_[predictionCount] << endl;

                                // How many prediction have been made
                                predictionCount = predictionCount + 1;
                                // Switch to gather displacement fields
                                predictionSwitch = true;
                                // Iteration to start collecting data
                                startIter = iCorr;
                                endIter = iCorr + machinePredictorIter_;
                                Info << "Starting iteration is: " << startIter << endl;
                                Info << "Ending iteration is: " << endIter << endl;
                                Info << nl << "Will collect residuals from iCorr: " << startIter << " to " << endIter - 1 << endl;
                            }

                        }
                    }


                    if (predictionSwitch)
                    {
                        // Store previous D fields
                        // SL 6/6/23: 
                        // iCorr > startIter - 1  Is to ALLOW FOR CUURENT ITERATION TO BE USE

                        if (iCorr >= startIter  && iCorr < endIter) 
                        {
                            Info << "Adding D from iteration " << iCorr << " for residual " << residual_print << endl;
                            index = iCorr - startIter;
                            Info << "Index: " << index << endl;

                            storeMLInputFields();

                            // if (machineLearning_)
                            // {
                            //     prevCellD_[iCorr - startIter] = D(); 

                            //     if (useResiduals_)
                            //     {
                            //         updateResidualD();
                            //         prevCellR_[iCorr - startIter] = residualD; 
                            //     }  
                            // }
                            // Only need to write fields ifnot machine learning
                            // else
                            // {
                            //     // write D fields to file
                            //     writeDisplacementIteration(predictionCount, iCorr - startIter, false);
                                
                            //     // write R fields to file         
                            //     if (useResiduals_)
                            //     {
                            //         writeResidualIteration(predictionCount, iCorr - startIter, false);
                            //     }
                            // }                          
                        }
                    }

                    // Store fields for under-relaxation and residual calculation
                    D().storePrevIter();

                    // if (iCorr == machinePredictorIter_ + startIter -1 && predictionSwitch)
                    if (iCorr == machinePredictorIter_ + startIter && predictionSwitch)
                    {
                        if (testConverged_)
                        {
                            updateD_testConverged();
                            //    updateSigma_testConverged();    // Doesnt make a difference due to correction
                            
                            //  Note: this stores D.prevIter
                            BoundaryTractionLoop();
                        }

                        if (machineLearning_)
                        {
                            Info << "iCorr: " << iCorr << endl;
                            Info << "residual_print: " << residual_print << endl;
                            Info << "predictionCount: " << predictionCount << endl;

                            // Predict displacement
                            updateD_ML(predictionCount, jsonFiles_[predictionCount - 1], relaxD_ML_);
                            // Boundary traction loop
                            BoundaryTractionLoop();        
                        }

                        // Turn off prediction switch
                        predictionSwitch = false;
                    }

                    // Linear momentum equation total displacement form
                    fvVectorMatrix DEqn(
                        rho() * fvm::d2dt2(D()) == 
                        fvm::laplacian(impKf_, D(), "laplacian(DD,D)")
                        - fvc::laplacian(impKf_, D(), "laplacian(DD,D)") 
                        + fvc::div(sigma(), "div(sigma)") + rho() * g()
                        + stabilisation().stabilisation(D(), gradD(), impK_)
                    );

                    // Info << fvc::div(sigma(), "div(sigma)") + rho() * g() << endl;

                    // Under-relaxation the linear system
                    DEqn.relax();

                    // Enforce any cell displacements
                    solidModel::setCellDisps(DEqn);

                    // Hack to avoid expensive copy of residuals
#ifdef OPENFOAMESI
                    const_cast<dictionary &>(mesh().solverPerformanceDict()).clear();
#endif

                    // Solve the linear system
                    solverPerfD = DEqn.solve();


                    // Relax fields when not predicting
                    if (!machineLearning_)
                    {
                        // Fixed or adaptive field under-relaxation
                        relaxField(D(), iCorr);
                    }
                    else if (machineLearning_ && (iCorr != machinePredictorIter_))
                    {
                        // Fixed or adaptive field under-relaxation
                        relaxField(D(), iCorr);
                    }

                    // Update increment of displacement
                    DD() = D() - D().oldTime();

                    // Update gradient of displacement
                    mechanical().grad(D(), gradD());

                    // Update gradient of displacement increment
                    gradDD() = gradD() - gradD().oldTime();

                    // Update the momentum equation inverse diagonal field
                    // This may be used by the mechanical law when calculating the
                    // hydrostatic pressure
                    const volScalarField DEqnA("DEqnA", DEqn.A());

                    // Calculate the stress using run-time selectable mechanical law
                    mechanical().correct(sigma());

                    // Update impKf to improve convergence
                    // Note: impK and rImpK are not updated as they are used for
                    // traction boundaries
                    // if (iCorr % 10 == 0)
                    //{
                    //    impKf_ = mechanical().impKf();
                    //}

                    // Write D field for each iteration
                    // if (predictionSwitch)
                    // {
                    //     writeDisplacementIteration(predictionCount, iCorr - startIter, false);
                    // }

                    // // Store previous D fields
                    // if (iCorr > startIter && iCorr < prevCellD_.size() + 1 + startIter)
                    // {
                    //     prevCellD_[iCorr - 1 - startIter] = D();
                    // }

                    if (writeIndivResidualFile_)
                    {
                        writeResidualFile(iCorr);
                    }

                    if (writeCellDisplacement_)
                    {
                        writeCellDisplacementList(iCorr);
                    }
                   
                    if (debugFields_)
                    {

                        // if ((iCorr > 0) & (iCorr < 60))
                        // {
                            writeDebugFields(iCorr);
                        // }

                        // if ((iCorr > 128) & (iCorr < 160))
                        // {
                            // writeDebugFields(iCorr);
                        // }

                    }

                } while (
                    !converged(
                        iCorr,
#ifdef OPENFOAMESIORFOUNDATION
                        mag(solverPerfD.initialResidual()),
                        cmptMax(solverPerfD.nIterations()),
#else
                        solverPerfD.initialResidual(),
                        solverPerfD.nIterations(),
#endif
                        D()) &&
                    ++iCorr < nCorr());

                // Write final iteration displacement
                for (int i = 1; i < noPredictions_ + 2; i++)
                {
                    writeDisplacementIteration(i, iCorr, false);
                    writeResidualIteration(i, iCorr, false);
                }
                writeDisplacementIteration(0, iCorr, true);
                writeResidualIteration(0, iCorr, true);


                // Interpolate cell displacements to vertices
                mechanical().interpolate(D(), pointD());

                // Increment of displacement
                DD() = D() - D().oldTime();

                // Increment of point displacement
                pointDD() = pointD() - pointD().oldTime();

                // Velocity
                U() = fvc::ddt(D());
            } while (mesh().update());

#ifdef OPENFOAMESIORFOUNDATION
            SolverPerformance<vector>::debug = 1;
#else
            blockLduMatrix::debug = 1;
#endif

            return true;
        }

        tmp<vectorField> linGeomTotalDispSolidML::tractionBoundarySnGrad(
            const vectorField &traction,
            const scalarField &pressure,
            const fvPatch &patch) const
        {

            // Patch index
            const label patchID = patch.index();

            // Patch mechanical property
            const scalarField &impK = impK_.boundaryField()[patchID];

            // Patch reciprocal implicit stiffness field
            const scalarField &rImpK = rImpK_.boundaryField()[patchID];

            // Patch gradient
            const tensorField &pGradD = gradD().boundaryField()[patchID];

            // const fvPatchField<tensor>& gradU =
            // patch().lookupPatchField<volTensorField, tensor>("grad(U)")

            // Patch stress
            const symmTensorField &pSigma = sigma().boundaryField()[patchID];

            // Patch unit normals
            const vectorField n(patch.nf());    

            // Patch mechanical property
            const scalarField& mu = muf_.boundaryField()[patchID];
            const scalarField& lambda = lambdaf_.boundaryField()[patchID];

            vectorField nSigma = (
                                    (n & (mu*pGradD))
                                    + (n & (mu*pGradD.T()))
                                    + (n*lambda*tr(pGradD))
                                );
            vectorField trial2 = (
                                    n & pSigma
                                );

            // Info  << nSigma[0] << endl;
            // Info  << trial2[0] << endl;


            // return tmp<vectorField>(
            //     new vectorField(
            //         (
            //             (traction - n * pressure)
            //             - nSigma
            //             + (n & impK * pGradD)
            //         )*
            //         rImpK));

            return tmp<vectorField>(
                new vectorField(
                    (
                        (traction - n * pressure) - (n & (pSigma - impK * pGradD))) *
                    rImpK));


            // return tmp<vectorField>(
            //     new vectorField
            //         (
            //             (
            //                 (traction - n*pressure)
            //                 - (
            //                     n & 
            //                     (
            //                         (mu*pGradD.T())
            //                         - ((mu + lambda)*pGradD)
            //                     )
            //                     - (n*lambda*tr(pGradD))
            //                 )
            //             )*rImpK;
            //         )
            // );

            // return tmp<vectorField>(
            //     new vectorField(
            //         (
            //             (traction - n * pressure) - (n & (pSigma - impK * pGradD))) *
            //         rImpK));

            // Return patch snGrad
            // return tmp<vectorField>
            // (
            //     new vectorField
            //     (
            //         (
            //             (traction - n*pressure)
            //         - (n & (pSigma - (2.0*mu + lambda)*pGradD))
            //         )/(2.0*mu + lambda)
            //     )
            // );

            // gradient() =
            // (
                // (traction_ - (pressure_)*n)
                // - (n & (mu.value()*gradU.T() - (mu.value() + lambda.value())*gradU))
                // - n*lambda.value()*tr(gradU)
            // )/(2.0*mu.value() + lambda.value())
        }

        void linGeomTotalDispSolidML::updateD_ML(int predictionCount, fileName jsonFile, scalar relaxD_ML_)
        {
            Info << nl << "Using machine learning to predict D field" << nl << endl;

            Info << "Prediction number: " << predictionCount << endl;

            // Load frugally-deep model
            Info << "Json file location is: " << jsonFile << endl; // move outside of loop

            // // Create the keras model
            auto model = fdeep::load_model(jsonFile);

            Info << "Keras model loaded" << endl;

            forAll(D(), cellI)
            {
                int inputVectorSize;
                inputVectorSize = inputSize*machinePredictorIter_;

                std::vector<double> inputFoam(inputVectorSize, 0);
                std::vector<double> scaledInput(inputVectorSize, 0);

                // Loop through iteration
                forAll(storedInputFields, iterI)
                {

                    // if (cellI==0){

                    //     Info << "Iteration: " << iterI << endl;

                    // }

                    //  Loop through input features
                    forAll(storedInputFields[iterI], inputF)
                    {

                        // if (cellI==0){

                        //     Info << "inputF: " << inputF << endl;

                        // }

                        // if (cellI==0){
                        //     Info << "Index in inputFoam: " << iterI*inputSize + inputF << endl;
                        // }

                        inputFoam[iterI*inputSize + inputF] = storedInputFields[iterI][inputF][cellI]; 

                        // Info << "scaling index:" << endl;

                        // Info << iterI + (predictionCount-1)*(machinePredictorIter_+1) << endl;

                        // scale, input after python code is written
                        scaledInput[iterI*inputSize + inputF] = 
                        scale
                        (
                            inputFoam[iterI*inputSize + inputF],
                            kerasScalingMeans_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputF], 
                            kerasScalingStds_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputF]
                        );
                       
                    }
                }

                // Info << "inputFoam: " <<  endl;

                // forAll(inputFoam, i)
                // {
                //     Info << inputFoam[i];
                // }


                const auto prediction = model.predict //_single_output
                                        (
                                            {fdeep::tensor(
                                                fdeep::tensor_shape(static_cast<std::size_t>(inputVectorSize)),
                                                scaledInput)});  

                int prediction_index;
                prediction_index = machinePredictorIter_ + machinePredictorIter_*(predictionCount-1) + (predictionCount-1);

                std::vector<double> result;
                std::vector<double> scaledResult;
                if(predictZ_)
                {         

                    scaledResult =
                        {
                            prediction[0].get(fdeep::tensor_pos(0)),
                            prediction[0].get(fdeep::tensor_pos(1)),
                            prediction[0].get(fdeep::tensor_pos(2))
                            };
                    
                    result =
                        {
                            invScale(scaledResult[0], kerasScalingMeans_[prediction_index][0], kerasScalingStds_[prediction_index][0]),
                            invScale(scaledResult[1], kerasScalingMeans_[prediction_index][1], kerasScalingStds_[prediction_index][1]),
                            invScale(scaledResult[2], kerasScalingMeans_[prediction_index][2], kerasScalingStds_[prediction_index][2]),
                        };
                }
                else
                {
                    scaledResult =
                        {
                            prediction[0].get(fdeep::tensor_pos(0)),
                            prediction[0].get(fdeep::tensor_pos(1)),
                            };
                    
                    result =
                        {
                            invScale(scaledResult[0], kerasScalingMeans_[prediction_index][0], kerasScalingStds_[prediction_index][0]),
                            invScale(scaledResult[1], kerasScalingMeans_[prediction_index][1], kerasScalingStds_[prediction_index][1]),
                        };
                }

                // Copy the result back into an OpenFOAM array

                if (cellI==0)
                {
                    Info << "Using ML relaxation factor: " << relaxD_ML_ << endl;
                }

                D()[cellI].x() = relaxD_ML_*result[0] +  (1-relaxD_ML_)*D()[cellI].x();
                D()[cellI].y() = relaxD_ML_*result[1] +  (1-relaxD_ML_)*D()[cellI].y();

                if (predictZ_)
                {
                    D()[cellI].z() = relaxD_ML_*result[2] +  (1-relaxD_ML_)*D()[cellI].z();
                }            
            }

            Info << "Finished prediction" << endl;

            writePredictedDField(predictionCount);
        }

        void linGeomTotalDispSolidML::updateD_testConverged()
        {
            Info << "Reading converged D values from " + convergedCase_ + runTime().timeName() << endl;

            vectorIOField convergedD(
                IOobject(
                    "convergedD_cells",                          // name of file on disk
                    convergedCase_ + "/" + runTime().timeName(), // where it is located (read from 0 at the start)
                    // runTime.constant(), // where it is located
                    mesh(), // object registry
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE)
                // vectorField(length, initVal) // only needed for NO_READ or READ_IF_PRESENT
            );

            forAll(D(), cellI)
            {

                D()
                [cellI].x() = convergedD[cellI][0];
                D()
                [cellI].y() = convergedD[cellI][1];
                D()
                [cellI].z() = convergedD[cellI][2];
            }
        }

        void linGeomTotalDispSolidML::updateSigma_testConverged()
        {
            Info << "Reading converged sigma values from " + convergedCase_ + "/" + runTime().timeName() << endl;

            volSymmTensorField convergedSigma(
                IOobject(
                    "sigma",                                     // name of file on disk
                    convergedCase_ + "/" + runTime().timeName(), // where it is located (read from 0 at the start)
                    // runTime.constant(), // where it is located
                    mesh(), // object registry
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                this->mesh() // vectorField(length, initVal) // only needed for NO_READ or READ_IF_PRESENT
            );

            Info << "convergedSigma" << convergedSigma << endl;

            forAll(sigma(), cellI)
            {

                sigma()[cellI] = convergedSigma[cellI];
                // sigma()[cellI] = convergedSigma[cellI];
                // sigma()[cellI] = convergedSigma[cellI];
            }

            // Info<< "D: "<< nl << D() << endl;

            Info << "Performing prediction corrections" << endl;
        }

        void linGeomTotalDispSolidML::BoundaryTractionLoop()
        {
            Info << "Start of Boundary loop" << endl;

            bool convergedBcTrac = false;

            int BCloopCorr = 0;
            scalar residual = 0;

            do
            {
                D().storePrevIter();

                mechanical().correct(sigma());   // Calculates sigma using grad D
                D().correctBoundaryConditions(); // Applies BCs to boundaries? (changes D)

                mechanical().grad(D(), gradD()); // Calculates new grad D

                residual =
                    max(
                        mag(D() - D().prevIter())
                        ).value() 
                        /
                    (max
                        (mag(D())).value() + SMALL // SMALL = 1e-15 (defined by OpenFOAM) - To avoid dividing by zero
                    );
                    

                if (residual < tractionBCtol_)
                {
                    convergedBcTrac = true;
                }

                BCloopCorr = BCloopCorr + 1;

                // Info << "BCloopCorr: " << BCloopCorr << endl;
                // Info << "residual: " << residual << endl;

            } while (!convergedBcTrac);

            Info << "BC correction loop took " << BCloopCorr << " iterations" << nl;

            BCloopCorrFile_() << runTime().timeName() << tab << residual << tab << BCloopCorr << nl;

            
        }

        void linGeomTotalDispSolidML::writeDisplacementIteration(int predictionCount, int iter, bool converged)
        {
            if (iter == 0)
            {
                Foam::mkDir(runTime().timeName() + "/" + Foam::name(predictionCount));
            }

            // Write D field for each iteration
            {

                fileName fName;
                std::string directory;

                if (!converged)
                {
                    fName = Foam::name(predictionCount) + "/solidsCellD_iteration" + Foam::name(iter);

                    if (predictionCount == 0)
                    {
                        fName = "/solidsCellD_iteration" + Foam::name(iter);
                    }
                }
                else if (converged)
                {
                    fName = "convergedD_cells";
                }

                vectorIOField dataToWrite(
                    IOobject(
                        fName,
                        runTime().timeName(),
                        runTime(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE),
                    D());
                dataToWrite.write();

                // volVectorField dataToWrite(

                // IOobject(
                //     fName,
                //     runTime().timeName(),
                //     runTime(),
                //     IOobject::NO_READ,
                //     IOobject::AUTO_WRITE),
                // D());

                    
                // dataToWrite.write();
            }
        }

        void linGeomTotalDispSolidML::writeResidualIteration(int predictionCount, int iter, bool converged)
        {
            updateResidualD();

            if (iter == 0)
            {
                Foam::mkDir(runTime().timeName() + "/" + Foam::name(predictionCount));
            }

            // Write R field for each iteration
            {

                fileName fName;
                std::string directory;

                if (!converged)
                {
                    fName = Foam::name(predictionCount) + "/residualD_iteration" + Foam::name(iter);

                    if (predictionCount == 0)
                    {
                        fName = "/residualD_iteration" + Foam::name(iter);
                    }
                }
                else if (converged)
                {
                    fName = "convergedR_cells";
                }

                vectorIOField dataToWrite(
                    IOobject(
                        fName,
                        runTime().timeName(),
                        runTime(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE),
                    residualD);
                dataToWrite.write();
            }
        }

        void linGeomTotalDispSolidML::writeDebugFields(int iter)
        {
            if (Foam::exists(Foam::name(iter)))
            {
                Info << "directory exists" << endl;
            }
            else
            {
                Info << "Creating directory: " << iter << " to store fields" << endl;
                Foam::mkDir(Foam::name(iter));
            }

            //  Displacement
            if (iter > 0)
            {

                volVectorField dataToWrite(
                IOobject(
                    "D",
                    runTime().timeName(),
                    "../" + Foam::name(iter),
                    mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE),

                D());
                // dataToWrite.write();
            }

            //  Error
            Info << "Reading D_solution\n" << endl;
            volVectorField D_solution
            (
                IOobject
                (
                    "D_solution",
                    "0",
                    mesh(),
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh()
            );

            Info << "Calculating and writing error" << endl;
            volVectorField error_D
            (
                IOobject(
                    "error_D",
                    runTime().timeName(),
                    "../" + Foam::name(iter),
                    mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                    ),
                D()
            );

            vectorField diff_D = D() - D_solution;

            forAll(error_D, cell)
            {
                error_D[cell][0] = sqrt(pow(diff_D[cell][0], 2));
                error_D[cell][1] = sqrt(pow(diff_D[cell][1], 2));
                error_D[cell][2] = sqrt(pow(diff_D[cell][2], 2));
            };

            // error_D.write();

            // Relative residuals
            // This is a normalised scalar field
            scalar denom = gMax(
                Field<scalar>(
                    mag(D().internalField() - D().oldTime().internalField())));
            if (denom < SMALL)
            {
                denom = max(
                    gMax(
                        mag(D().internalField())),
                    SMALL);
            }

            scalarField relativeresidual =
                mag(D().internalField() - D().prevIter().internalField())/
                denom;  

            volScalarField relresidual
            (
                IOobject
                (
                    "relresidual",
                    runTime().timeName(),
                    "../" + Foam::name(iter),
                    mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh(),
                dimensionedScalar ("relresidual", dimensionSet(0,0,0,0,0,0,0), 0.0)
            );

            relresidual.internalField() = relativeresidual;
            // relresidual.write();


            // if (iter > 0)
            // {   
            //     dataToWrite.write();
            // }
            // error_D.write();
            // relresidual.write();

            errorDcellsFile_() << iter << tab;
            DcellsFile_() << iter << tab;

            forAll(debug_cells, cellI)
            {
                errorDcellsFile_() << error_D[debug_cells[cellI]][0] << tab;
                errorDcellsFile_() << error_D[debug_cells[cellI]][1] << tab;
                errorDcellsFile_() << error_D[debug_cells[cellI]][2] << tab;

                DcellsFile_() << D().internalField()[debug_cells[cellI]][0] << tab;
                DcellsFile_() << D().internalField()[debug_cells[cellI]][1] << tab;
                DcellsFile_() << D().internalField()[debug_cells[cellI]][2] << tab;

            }
            errorDcellsFile_() << endl;
            DcellsFile_() << endl;
        }

        void linGeomTotalDispSolidML::writePredictedDField(int predictionCount)
        {
            // volVectorField D_predicted = D()*1;

            fileName fName = "D_predicted_" + Foam::name(predictionCount);

            // volVectorField D_predicted_write(

            //     IOobject(
            //         fName,
            //         runTime().timeName(),
            //         runTime(),
            //         IOobject::NO_READ,
            //         IOobject::AUTO_WRITE),
            //     D_predicted);
                
            // D_predicted_write.write();

            volVectorField D_predicted_write(
                    IOobject(
                        fName,
                        runTime().timeName(),
                        runTime(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE),
                    D());
            D_predicted_write.write();
        }

        // Writes individual residual for each direction
        void linGeomTotalDispSolidML::writeResidualFile(int iteration)
        {

            residualD = D() - D().prevIter();

            auto res_x = scalarField(residualD.size(), 0);
            auto res_y = scalarField(residualD.size(), 0);
            auto res_z = scalarField(residualD.size(), 0);

            forAll(res_x, cell)
            {
                res_x[cell] = sqrt(pow(residualD[cell][0], 2));
                res_y[cell] = sqrt(pow(residualD[cell][1], 2));
                res_z[cell] = sqrt(pow(residualD[cell][2], 2));
            };

            auto denom_x = scalarField(D().size(), 0);
            auto denom_y = scalarField(D().size(), 0);
            auto denom_z = scalarField(D().size(), 0);

            forAll(denom_x, cell)
            {
                denom_x[cell] = sqrt(pow(D()[cell][0], 2));
                denom_y[cell] = sqrt(pow(D()[cell][1], 2));
                denom_z[cell] = sqrt(pow(D()[cell][2], 2));
            };

            scalar res_x_final = max(res_x) / (max(denom_x) + SMALL);
            scalar res_y_final = max(res_y) / (max(denom_y) + SMALL);
            scalar res_z_final = max(res_z) / (max(denom_z) + SMALL);

            indivResidualFilePtr_() << res_x_final << tab << res_y_final << tab << res_z_final << endl;
        
            scalarField res_x_field = res_x / (max(denom_x) + SMALL);
            scalarField res_y_field = res_y / (max(denom_y) + SMALL);
            scalarField res_z_field = res_z / (max(denom_z) + SMALL);

            volVectorField residual_D = D() * 1;

            forAll(residual_D, cell)
            {
                residual_D[cell][0] = res_x_field[cell];
                residual_D[cell][1] = res_y_field[cell];
                residual_D[cell][2] = res_z_field[cell];
            };

            fileName fName;

            fName = "residual_D_iteration" + Foam::name(iteration);

            volVectorField residual_D_write(
                IOobject(
                    fName,
                    runTime().timeName(),
                    runTime(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE),
                residual_D);
                    
            residual_D_write.write();

        }

        // Writes individual residual for each direction
        void linGeomTotalDispSolidML::updateResidualD()
        {

            residualD = D() - D().prevIter();

            auto res_x = scalarField(residualD.size(), 0);
            auto res_y = scalarField(residualD.size(), 0);
            auto res_z = scalarField(residualD.size(), 0);

            forAll(res_x, cell)
            {
                res_x[cell] = sqrt(pow(residualD[cell][0], 2));
                res_y[cell] = sqrt(pow(residualD[cell][1], 2));
                res_z[cell] = sqrt(pow(residualD[cell][2], 2));
            };

            auto denom_x = scalarField(D().size(), 0);
            auto denom_y = scalarField(D().size(), 0);
            auto denom_z = scalarField(D().size(), 0);

            forAll(denom_x, cell)
            {
                denom_x[cell] = sqrt(pow(D()[cell][0], 2));
                denom_y[cell] = sqrt(pow(D()[cell][1], 2));
                denom_z[cell] = sqrt(pow(D()[cell][2], 2));
            };
       
            scalarField res_x_field = res_x / (max(denom_x) + SMALL);
            scalarField res_y_field = res_y / (max(denom_y) + SMALL);
            scalarField res_z_field = res_z / (max(denom_z) + SMALL);

            forAll(residualD, cell)
            {
                residualD[cell][0] = res_x_field[cell];
                residualD[cell][1] = res_y_field[cell];
                residualD[cell][2] = res_z_field[cell];
            };
        }

        // Writes individual residual for each direction
        scalarField linGeomTotalDispSolidML::scalarFieldrelResidual()
        {
            scalar denom = gMax(
            Field<scalar>(
                    mag(D().internalField() - D().oldTime().internalField())));
            if (denom < SMALL)
            {
                denom = max(
                    gMax(
                        mag(D().internalField())),
                    SMALL);
            }

            scalarField relativeresidual =
                mag(D().internalField() - D().prevIter().internalField())/
                denom;  

            return relativeresidual;
        }

        void linGeomTotalDispSolidML::writeCellDisplacementList(int iteration)
        {

            if (Pstream::master())
            {
                forAll(cellList_, cell)
                {
                    cellDisplacementFile_() << iteration << tab << D()[cellList_[cell]][0] << tab << D()[cellList_[cell]][1] << tab << D()[cellList_[cell]][2] << tab;
                };

                cellDisplacementFile_() << endl;
            }
        }

        scalar linGeomTotalDispSolidML::residualvf()
        {
            // Calculate displacement residual based on the relative change of vf
            scalar denom = gMax(
                Field<scalar>(
                    mag(D().internalField() - D().oldTime().internalField())));
            if (denom < SMALL)
            {
                denom = max(
                    gMax(
                        mag(D().internalField())),
                    SMALL);
            }
            const scalar residualvf =
                gMax(
                    mag(D().internalField() - D().prevIter().internalField())) /
                denom;

            return residualvf;
        }

        void linGeomTotalDispSolidML::initMLFieldStoring()
        {
            inputSize = 0;
            // fileName fName;

            Info << "Initialising input fields" << endl;

            Info << "inputsML_: " << inputsML_ << endl;

            int count = 0;
            forAll(inputsML_, iterI)
            {
                // D fields
                if (inputsML_[iterI] == "D")
                {
                    if (predictZ_)
                    {
                        inputSize = inputSize + 3;
                        inputSizes[iterI] = 3;
                    }
                    else 
                    {
                        inputSize = inputSize + 2;
                        inputSizes[iterI] = 2;
                    }
                }
                // Individual relative residual
                if (inputsML_[iterI] == "individualRelativeResidual")
                {
                    if (predictZ_)
                    {
                        inputSize = inputSize + 3;
                        inputSizes[iterI] = 3;
                    }
                    else 
                    {
                        inputSize = inputSize + 2;
                        inputSizes[iterI] = 2;
                    }
                }
                // Relative residual
                if (inputsML_[iterI] == "relativeResidual")
                {
                    inputSize = inputSize + 1;
                    inputSizes[iterI] = 1;
                }
                //  Coordinates
                if (inputsML_[iterI] == "coordinates")
                {
                    if (predictZ_)
                    {
                        inputSize = inputSize + 3;
                        inputSizes[iterI] = 3;
                    }
                    else 
                    {
                        inputSize = inputSize + 2;
                        inputSizes[iterI] = 2;
                    }
                }
                //  gradD
                if (inputsML_[iterI] == "gradD")
                {
                    if (predictZ_)
                    {
                        inputSize = inputSize + 9;  
                        inputSizes[iterI] = 9;       
                    }
                    else 
                    {
                        inputSize = inputSize + 6; 
                        inputSizes[iterI] = 6;      
                    }
                }
            } 

            Info << "inputSizes: " << inputSizes << endl;
            Info << "ML input feature has a size of : " << inputSize << endl;

            // Make a List of a PtrList of scalarFields
            storedInputFields.setSize(machinePredictorIter_);

            forAll(storedInputFields, iterI)
            {
 
                PtrList<scalarField> tempScalarFieldList;

                tempScalarFieldList.setSize(inputSize);

                forAll(tempScalarFieldList, iterJ)
                {
                   tempScalarFieldList.set(
                        iterJ,
                        new scalarField(D().internalField().size(), 0.0) 
                    );
                }

                storedInputFields[iterI] = tempScalarFieldList;
            }
            count = count + 1;
        }

        void linGeomTotalDispSolidML::storeMLInputFields()
        {
            // Info << "gradD(): " << gradD() << endl;

            int feature_count = 0;

            //  Loop through each input size
            forAll(inputSizes, input)
            {

                if (inputsML_[input]=="D")
                {
                    //  acces how many dimensions
                    int dim = inputSizes[input];
                   
                    //  Loop through each dimension creating a scalar field
                    for (int i = 0; i < dim; i++)
                    {
                        scalarField temp = scalarField(D().internalField().size(), 0.0);
                        forAll(temp, cellI)
                        {
                            temp[cellI] = D()[cellI][i];
                        }

                        // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
                        storedInputFields[index][feature_count + i] = temp;

                        // Maybe write should be here? 
                    }            
                    
                    feature_count = feature_count + dim;
                              
                }

                if (inputsML_[input]=="individualRelativeResidual")
                {
                    //  acces how many dimensions
                    int dim = inputSizes[input];
                    
                    //  Loop through each dimension creating a scalar field
                    for (int i = 0; i < dim; i++)
                    {
                        scalarField temp = scalarField(D().internalField().size(), 0.0);

                        updateResidualD();
                        forAll(temp, cellI)
                        {
                            temp[cellI] = residualD[cellI][i];
                        }
                        // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
                        storedInputFields[index][feature_count + i] = temp;

                        // Maybe write should be here? 
                    }            
                    
                    feature_count = feature_count + dim;
           
                }
                
                if (inputsML_[input]=="relativeResidual")
                {
                    //  acces how many dimensions
                    int dim = inputSizes[input];
                    
                    //  Loop through each dimension creating a scalar field
                    for (int i = 0; i < dim; i++)
                    {
                        scalarField temp = scalarFieldrelResidual();

                        // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
                        storedInputFields[index][feature_count + i] = temp;

                        // Maybe write should be here? 
                    }            
                    
                    feature_count = feature_count + dim;              
                }

                if (inputsML_[input]=="coordinates")
                {
                    //  acces how many dimensions
                    int dim = inputSizes[input];
                    
                    //  Loop through each dimension creating a scalar field
                    for (int i = 0; i < dim; i++)
                    {
                        scalarField temp = scalarField(D().internalField().size(), 0.0);

                        forAll(temp, cellI)
                        {
                            temp[cellI] = mesh().C()[cellI][i];
                        }
                        // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
                        storedInputFields[index][feature_count + i] = temp;

                        // Maybe write should be here? 
                    }            
                    
                    feature_count = feature_count + dim;                  
                }

                if (inputsML_[input]=="gradD")
                {
                    //  acces how many dimensions
                    int dim = inputSizes[input];
                    
                    //  Loop through each dimension creating a scalar field
                    for (int i = 0; i < dim; i++)
                    {
                        scalarField temp = scalarField(D().internalField().size(), 0.0);
                        forAll(temp, cellI)
                        {
                            temp[cellI] = gradD()[cellI][i];
                        }

                        // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
                        storedInputFields[index][feature_count + i] = temp;

                        // Maybe write should be here? 
                    }            
                    
                    feature_count = feature_count + dim;                     
                }
            }

            for (int inputFeature = 0; inputFeature < inputSize; inputFeature++)
            {
                // Write storedInputFields[index] here
                if(writeFields_)
                {
                    if (index == 0)
                    {
                        Foam::mkDir(runTime().timeName() + "/" + Foam::name(predictionCount));
                    }

                    if (inputFeature == 0)
                    {
                        Foam::mkDir(runTime().timeName() + "/" + Foam::name(predictionCount) + "/" + Foam::name(index));
                    }

                    fileName fName;

                    fName = Foam::name(predictionCount) + "/" + Foam::name(index) + "/inputFeature_" + Foam::name(inputFeature);

                    scalarIOField dataToWrite(
                        IOobject(
                            fName,
                            runTime().timeName(),
                            runTime(),
                            IOobject::NO_READ,
                            IOobject::AUTO_WRITE),
                        storedInputFields[index][inputFeature]
                        );
                    dataToWrite.write();

                }
            }
        }


        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    } // End namespace solidModels

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
