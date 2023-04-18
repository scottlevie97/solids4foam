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
#include "momentumStabilisation.H"
#define FDEEP_FLOAT_TYPE double
#include <fdeep/fdeep.hpp>

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
              predictor_(solidModelDict().lookupOrDefault<Switch>("predictor", false)),
              machineLearning_(
                  solidModelDict().lookupOrDefault<Switch>("machineLearning", Switch(false))),
              machinePredictorIter_(
                  solidModelDict().lookupOrDefault<scalar>("iterationToApplyMachineLearningPredictor", 11)),
              jsonFile_(
                  solidModelDict().lookupOrDefault<fileName>("jsonFile", "jsonFile")),
              convergedCase_(
                  solidModelDict().lookupOrDefault<fileName>("convergedCase", "../plateHoleTotalDisp")),
              prevCellD_(),
              testConverged_(
                  solidModelDict().lookupOrDefault<Switch>("testConverged", Switch(false))),
              tracBcIter_(
                  solidModelDict().lookupOrDefault<scalar>("tracBcIter", 50)),
              BCloopCorrFile_(),
              writeDisplacementField_(
                  solidModelDict().lookupOrDefault<Switch>("writeDisplacementField", Switch(false))),
              writeDisplacementLimit_(
                  solidModelDict().lookupOrDefault<scalar>("writeDisplacementLimit", 20))
        {
            DisRequired();

            // Force all required old-time fields to be created
            fvm::d2dt2(D());

            // For consistent restarts, we will calculate the gradient field
            D().correctBoundaryConditions();
            D().storePrevIter();
            mechanical().grad(D(), gradD());

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

            if (machineLearning_)
            {

                // Load Scaling values
                kerasScalingMeans_ = List<vector>(
                    solidModelDict().lookup("kerasScalingMeans"));

                kerasScalingStds_ = List<vector>(
                    solidModelDict().lookup("kerasScalingStds"));

                // Determine size of D()   (Probably could be better)
                int Dsize = 0;
                forAll(D(), cellI)
                {
                    Dsize = Dsize + 1;
                }

                prevCellD_.setSize(machinePredictorIter_);
                forAll(prevCellD_, iterI)
                {
                    prevCellD_.set(
                        iterI,
                        new vectorField(Dsize, vector::zero) // hard coding workaround
                    );
                }
            }

            // Initiate BC Correction Loop File

            if (Pstream::master())
            {
                Info << "Creating BCloopCorr.dat" << endl;
                BCloopCorrFile_.set(
                    new OFstream(runTime.path() / "BCloopCorr.dat"));
                BCloopCorrFile_()
                    << "Time" << tab << "Converged residual " << tab << "BCloopCorr" << endl;
            }

            // OFstream BCloopCorrFile("BCloopCorr.dat");
            // BCloopCorrFile << "Time" << tab << "Converged residual "<< tab <<  "BCloopCorr" << endl;
        }

        // * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

        bool linGeomTotalDispSolidML::evolve()
        {
            Info << "Evolving solid solver" << endl;

            if (predictor_)
            {
                predict();
            }

            // Mesh update loop
            do
            {
                int iCorr = 0;
#ifdef OPENFOAMESIORFOUNDATION
                SolverPerformance<vector> solverPerfD;
                SolverPerformance<vector>::debug = 0;
#else
                lduSolverPerformance solverPerfD;
                blockLduMatrix::debug = 0;
#endif

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

                Info << "Solving the momentum equation for D" << endl;

                // Momentum equation loop
                do
                {

                    // Store previous D fields
                    if (machineLearning_ && iCorr > 0 && iCorr < prevCellD_.size() + 1)
                    {
                        prevCellD_[iCorr - 1] = D();
                    }

                    // Store fields for under-relaxation and residual calculation
                    D().storePrevIter();

                    if (iCorr == machinePredictorIter_)
                    {
                        if (testConverged_)
                        {
                            updateD_testConverged();
                            //    updateSigma_testConverged();    // Doesnt make a difference due to correction
                            BoundaryTractionLoop();
                        }

                        if (machineLearning_)
                        {
                            // Predict displacement
                            updateD_ML();
                            // Boundary traction loop
                            BoundaryTractionLoop();
                        }
                    }

                    // Linear momentum equation total displacement form
                    fvVectorMatrix DEqn(
                        rho() * fvm::d2dt2(D()) == fvm::laplacian(impKf_, D(), "laplacian(DD,D)") - fvc::laplacian(impKf_, D(), "laplacian(DD,D)") + fvc::div(sigma(), "div(sigma)") + rho() * g() + stabilisation().stabilisation(D(), gradD(), impK_));

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
                    if (writeDisplacementField_ && (iCorr < writeDisplacementLimit_))
                    {
                        writeDisplacementIteration(iCorr, false);
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
                writeDisplacementIteration(iCorr, true);
                writeDisplacementIteration(iCorr, false);

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

            // Patch stress
            const symmTensorField &pSigma = sigma().boundaryField()[patchID];

            // Patch unit normals
            const vectorField n(patch.nf());

            // Return patch snGrad
            return tmp<vectorField>(
                new vectorField(
                    (
                        (traction - n * pressure) - (n & (pSigma - impK * pGradD))) *
                    rImpK));
        }

        void linGeomTotalDispSolidML::updateD_ML()
        {
            Info << nl << "Using machine learning to predict D field" << nl << endl;

            // Load frugally-deep model
            Info << "Json file location is: " << jsonFile_ << endl; // move outside of loop

            // // Create the keras model
            auto model = fdeep::load_model(jsonFile_);

            Info << "Keras model loaded" << endl;

            forAll(D(), cellI)
            {
                // Extra column for coordinates
                auto inputFoam = vectorField(machinePredictorIter_ + 1, vector::zero);

                const int inputVectorSize = (machinePredictorIter_ + 1) * 3;

                std::vector<double> scaledInput(
                    inputVectorSize, 0);

                forAll(prevCellD_, iterI)
                {
                    inputFoam[iterI] = prevCellD_[iterI][cellI]; // prevCellD_ contains a value for iteration 0 (which is full of zeros)

                    scaledInput[iterI * 3] = scale(inputFoam[iterI].x(), kerasScalingMeans_[iterI].x(), kerasScalingStds_[iterI].x());
                    scaledInput[iterI * 3 + 1] = scale(inputFoam[iterI].y(), kerasScalingMeans_[iterI].y(), kerasScalingStds_[iterI].y());
                    scaledInput[iterI * 3 + 2] = scale(inputFoam[iterI].z(), kerasScalingMeans_[iterI].z(), kerasScalingStds_[iterI].z());
                }

                // Add coordinate:
                int coord_loc = machinePredictorIter_;

                inputFoam[coord_loc] = mesh().C()[cellI];

                scaledInput[coord_loc * 3] = scale(inputFoam[coord_loc].x(), kerasScalingMeans_[coord_loc].x(), kerasScalingStds_[coord_loc].x());
                scaledInput[coord_loc * 3 + 1] = scale(inputFoam[coord_loc].y(), kerasScalingMeans_[coord_loc].y(), kerasScalingStds_[coord_loc].y());
                scaledInput[coord_loc * 3 + 2] = scale(inputFoam[coord_loc].z(), kerasScalingMeans_[coord_loc].z(), kerasScalingStds_[coord_loc].z());

                const auto prediction = model.predict //_single_output
                                        (
                                            {fdeep::tensor(
                                                fdeep::tensor_shape(static_cast<std::size_t>(inputVectorSize)),
                                                scaledInput)});

                const std::vector<double> scaledResult =
                    {
                        prediction[0].get(fdeep::tensor_pos(0)),
                        prediction[0].get(fdeep::tensor_pos(1)),
                        prediction[0].get(fdeep::tensor_pos(2))};

                const std::vector<double> result =
                    {
                        invScale(scaledResult[0], kerasScalingMeans_[machinePredictorIter_ + 1][0], kerasScalingStds_[machinePredictorIter_ + 1][0]),
                        invScale(scaledResult[1], kerasScalingMeans_[machinePredictorIter_ + 1][1], kerasScalingStds_[machinePredictorIter_ + 1][1]),
                        invScale(scaledResult[2], kerasScalingMeans_[machinePredictorIter_ + 1][2], kerasScalingStds_[machinePredictorIter_ + 1][2]),
                    };

                // Copy the result back into an OpenFOAM array
                D()
                [cellI].x() = result[0];
                D()
                [cellI].y() = result[1];
                // D()
                // [cellI].z() = result[2];

                Info << "inputFoam" << nl << inputFoam << endl;

                Info << nl << "scaledInput" << endl;
                forAll(scaledInput, i)
                {
                    Info << scaledInput[i] << " ";

                    if (i % 3 == 2)
                    {
                        Info << endl;
                    }
                }

                Info << nl << "scaledResult" << endl;
                forAll(scaledResult, i)
                {
                    Info << scaledResult[i] << endl;
                }

                Info << nl << "result" << endl;
                forAll(result, i)
                {
                    Info << result[i] << endl;
                }
            }
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

            Info << "Performing prediction corrections" << endl;
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
                        mag(
                            D() - D().prevIter()))
                        .value() /
                    (max(
                         mag(D()))
                         .value() +
                     SMALL // SMALL = 1e-15 (defined by OpenFOAM) - To avoid dividing by zero
                    );

                if (residual < solutionTol())
                {
                    convergedBcTrac = true;
                }

                BCloopCorr = BCloopCorr + 1;

            } while (!convergedBcTrac);

            Info << "BC correction loop took " << BCloopCorr << " iterations" << nl;

            BCloopCorrFile_() << runTime().timeName() << tab << residual << tab << BCloopCorr << nl;
        }

        void linGeomTotalDispSolidML::writeDisplacementIteration(int iteration, bool converged)
        {
            // Write D field for each iteration
            {

                fileName fName;

                if (converged)
                {
                    fName = "convergedD_cells";
                }
                else if (!converged)
                {
                    fName = "solidsCellD_iteration" + Foam::name(iteration);
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
            }
        }

        void linGeomTotalDispSolidML::writePredictedDField()
        {
            volVectorField D_predicted = D() * 1;

            volVectorField D_predicted_write(

                IOobject(
                    "D_predicted",
                    runTime().timeName(),
                    runTime(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE),
                D_predicted);
            D_predicted_write.write();
        }

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    } // End namespace solidModels

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
