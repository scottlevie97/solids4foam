/*---------------------------------------------------------------------------*\
License
    This file is part of solids4foam.

    solids4foam is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    solids4foam is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with solids4foam.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "nonLinGeomTotalLagTotalDispSolidML.H"
#include "fvm.H"
#include "fvc.H"
#include "fvMatrices.H"
#include "addToRunTimeSelectionTable.H"
#define FDEEP_FLOAT_TYPE double
#include <fdeep/fdeep.hpp>
#include "fvCFD.H" 
#include <Eigen/Dense>
#include <cmath>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace solidModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(nonLinGeomTotalLagTotalDispSolidML, 0);
addToRunTimeSelectionTable
(
    solidModel, nonLinGeomTotalLagTotalDispSolidML, dictionary
);


// * * * * * * * * * * *  Private Member Functions * * * * * * * * * * * * * //


void nonLinGeomTotalLagTotalDispSolidML::predict()
{
    Info<< "Linear predictor" << endl;

    // Predict D using the velocity field
    // Note: the case may be steady-state but U can still be calculated using a
    // transient method
    D() = D().oldTime() + U()*runTime().deltaT();

    // Update gradient of displacement
    mechanical().grad(D(), gradD());

    // Total deformation gradient
    F_ = I + gradD().T();

    // Inverse of the deformation gradient
    Finv_ = inv(F_);

    // Jacobian of the deformation gradient
    J_ = det(F_);

    // Calculate the stress using run-time selectable mechanical law
    mechanical().correct(sigma());
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

nonLinGeomTotalLagTotalDispSolidML::nonLinGeomTotalLagTotalDispSolidML
(
    Time& runTime,
    const word& region
)
:
    solidModel(typeName, runTime, region),
    F_
    (
        IOobject
        (
            "F",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh(),
        dimensionedTensor("I", dimless, I)
    ),
    Finv_
    (
        IOobject
        (
            "Finv",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        inv(F_)
    ),
    J_
    (
        IOobject
        (
            "J",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        det(F_)
    ),
    impK_(mechanical().impK()),
    impKf_(mechanical().impKf()),
    rImpK_(1.0/impK_),
    predictor_(solidModelDict().lookupOrDefault<Switch>("predictor", false)),
    machinePredictorIter_
    (
        solidModelDict().lookupOrDefault<scalar>("iterationToApplyMachineLearningPredictor", 20)
    ),
    machineLearning_
    (
        solidModelDict().lookupOrDefault<Switch>("machineLearning", Switch(false))
    ),
    jsonFiles_
    (
        solidModelDict().lookupOrDefault<List<fileName>>("jsonFiles", List<fileName>(1, "jsonFile"))
    ),
    relaxD_ML_
    (
        solidModelDict().lookupOrDefault<scalar>("relaxD_ML", 1.0)
    ),
    inputsML_
    (
        solidModelDict().lookupOrDefault<List<fileName>>("inputsML", List<fileName>(1, "D"))
    ),
    predictZ_
    (
        solidModelDict().lookupOrDefault<Switch>("predictZ", true)
    ),
    predictionResiduals_
    (
        (solidModelDict().lookup("predictionResiduals"))
    ),
    noPredictions_(
                  solidModelDict().lookupOrDefault<scalar>("noPredictions", 1)),
    writeFields_(
        solidModelDict().lookupOrDefault<Switch>("writeFields", Switch(true))),
    tractionBCtol_(
        solidModelDict().lookupOrDefault<scalar>("tractionBCtolerance", 1e-10)),
    BCloopCorrFile_(),
    testConverged_(
        solidModelDict().lookupOrDefault<Switch>("testConverged", Switch(false))),
    convergedCase_(
        solidModelDict().lookupOrDefault<fileName>("convergedCase", "../base")),
    offsetIter_(
                  solidModelDict().lookupOrDefault<scalar>("offsetIter", 0)),
    debugFields_(
        solidModelDict().lookupOrDefault<Switch>("debugFields", false)),
    useCoordinates_(
        solidModelDict().lookupOrDefault<Switch>("useCoordinates", false))
{
    DisRequired();

    // Force all required old-time fields to be created
    fvm::d2dt2(D());

    if (predictor_)
    {
        // Check ddt scheme for D is not steadyState
        const word ddtDScheme
        (
#ifdef OPENFOAMESIORFOUNDATION
            mesh().ddtScheme("ddt(" + D().name() +')')
#else
            mesh().schemesDict().ddtScheme("ddt(" + D().name() +')')
#endif
        );

        if (ddtDScheme == "steadyState")
        {
            FatalErrorIn(type() + "::" + type())
                << "If predictor is turned on, then the ddt(" << D().name()
                << ") scheme should not be 'steadyState'!" << abort(FatalError);
        }
    }

    // For consistent restarts, we will update the relative kinematic fields
    D().correctBoundaryConditions();
    if (restart())
    {
        Info << "Inside restart" << endl;
        DD() = D() - D().oldTime();
        mechanical().grad(D(), gradD());
        gradDD() = gradD() - gradD().oldTime();
        F_ = I + gradD().T();
        Finv_ = inv(F_);
        J_ = det(F_);

        gradD().storeOldTime();

        // Let the mechanical law know
        mechanical().setRestart();
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


bool nonLinGeomTotalLagTotalDispSolidML::evolve()
{
    Info<< "Evolving solid solver" << endl;

    if (predictor_)
    {
        predict();
    }

    int iCorr = 0;
#ifdef OPENFOAMESIORFOUNDATION
    SolverPerformance<vector> solverPerfD;
    SolverPerformance<vector>::debug = 0;
#else
    lduSolverPerformance solverPerfD;
    blockLduMatrix::debug = 0;
#endif

    Info<< "Solving the total Lagrangian form of the momentum equation for D"
        << endl;


    int startIter = offsetIter_;
    int endIter;
    predictionCount = 0;

    // Make prediction is switch is on
    bool predictionSwitch = false;

    // Initialise inputsSizes array
    inputSizes = List<scalar>(inputsML_.size(), 0.0);

    initMLFieldStoring();

    if (machineLearning_)
    {
        // Load Scaling values
        kerasScalingMeans_ = List<scalarField>(
            solidModelDict().lookup("kerasScalingMeans"));

        kerasScalingStds_ = List<scalarField>(
            solidModelDict().lookup("kerasScalingStds"));        
    }

    D().storePrevIter();

    // Initialising here to make BC loop work
    // fvVectorMatrix DEqn
    // (
    //     rho()*fvm::d2dt2(D())
    //     == fvm::laplacian(impKf_, D(), "laplacian(DD,D)")
    //     - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
    //     + fvc::div(J_*Finv_ & sigma(), "div(sigma)")
    //     + rho()*g()
    //     + stabilisation().stabilisation(DD(), gradDD(), impK_)
    // );


    // Momentum equation loop
    do
    {
        // scalar residual_print = residualvf();
        scalar residual_print;
        if (iCorr == 0)
        {
            residual_print = 1;
        }
        else
        {
            residual_print = solverPerfD.initialResidual();
        }

        // The first residual when iCorr=0 is 0
        // Using this method will never use the resiudal when iCorr=0
        // This is why the results will be different to linGeomTotatlDispSolidML/linGeomTotatlDispSolidML.C
        if (predictionCount < noPredictions_ )
        {
            if ((residual_print < predictionResiduals_[predictionCount]) && (iCorr > startIter)) // was 0 instead of startIter
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

        if (predictionSwitch)
        {
            if (iCorr >= startIter  && iCorr < endIter) 
            {
                Info << "Adding D from iteration " << iCorr << " for residual " << residual_print << endl;
                index = iCorr - startIter;
                Info << "Index: " << index << endl;

                storeMLInputFields();   

                if (iCorr == endIter-1)
                {
                    writeOutput("D_before_prediction_" + Foam::name(predictionCount), true);
                }                     
            }
        }        

        // Store fields for under-relaxation and residual calculation
        D().storePrevIter();

        if (iCorr == machinePredictorIter_ + startIter && predictionSwitch)
        {
            if (machineLearning_)
            {
                Info << "iCorr: " << iCorr << endl;
                Info << "res: " << residual_print << endl;
                Info << "predictionCount: " << predictionCount << endl;

                // Predict displacement

                updateD_ML(predictionCount, jsonFiles_[predictionCount - 1], relaxD_ML_);
                // Boundary traction loop
                // BoundaryTractionLoop(DEqn);
                BoundaryTractionLoop();              
            }

            if (testConverged_)
            {

                updateMatrixResidual(); 
                volVectorField matrixResidual_write3
                (
                    IOobject(
                        "matrixResidual_before",
                        runTime().timeName(),
                        "../" + Foam::name(iCorr),
                        mesh(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                        ),
                    D()
                );     
                matrixResidual_write3.internalField() = matrixResidual;
                matrixResidual_write3.write();

                // UNCOMMENT
                updateD_testConverged();

                updateMatrixResidual(); 
                volVectorField matrixResidual_write1
                (
                    IOobject(
                        "matrixResidual_after_update",
                        runTime().timeName(),
                        "../" + Foam::name(iCorr),
                        mesh(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                        ),
                    D()
                );     
                matrixResidual_write1.internalField() = matrixResidual;
                matrixResidual_write1.write();
               
                // //  Note: this stores D.prevIter
                // // BoundaryTractionLoop(DEqn);
                BoundaryTractionLoop();   

                updateMatrixResidual(); 
                volVectorField matrixResidual_write2
                (
                    IOobject(
                        "matrixResidual_after_bc_loop",
                        runTime().timeName(),
                        "../" + Foam::name(iCorr),
                        mesh(),
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                        ),
                    D()
                );     
                matrixResidual_write2.internalField() = matrixResidual;
                matrixResidual_write2.write();

                // // Finv_.write();
                // // J_.write();
                // // sigma().write();                     
       
                // // break;

                // Insert known D, sigma, J and Finv_ here

        //         volVectorField D_solution
        //         (
        //             IOobject
        //             (
        //                 "D_solution",
        //                 "solutions",
        //                 mesh(),
        //                 IOobject::MUST_READ,
        //                 IOobject::NO_WRITE
        //             ),
        //             mesh()
        //         );

        //         D() = D_solution;

        //         Info << "D before loop: " << D() << endl;
        //         Info << "sigma before loop: " << sigma() << endl;


        //         BoundaryTractionLoop();

        //         Info << "D after loading: " << D() << endl;

        //         Info << "sigma after loop: " << sigma() << endl;


        //         // forAll(D().boundaryField(), patchI)
        //         // {
        //         //     D().boundaryField()[patchI].refGradient() = D_solution.boundaryField()[patchI].refGradient()*1;

        //         //     Info << "BOUNDARY1 : " << D().boundaryField()[patchI] << endl;
        //         //     Info << "BOUNDARY2: " << D_solution.boundaryField()[patchI] << endl;
        //         // }

        //         D().storePrevIter();

        //         // Info << "D after loading: " << D() << endl;

        //         // mechanical().grad(D(), gradD());

        //         Info << "gradD():" << gradD()[0] << endl;

        //         // volTensorField newgradD;

        //         // newgradD = fvc::grad(D());

        //         // Info << "newgradD:" << newgradD[0] << endl;

        //         // volTensorField newgradD(
        //         // IOobject(
        //         //         "newgradD",
        //         //         runTime().timeName(),
        //         //         runTime(),
        //         //         IOobject::NO_READ,
        //         //         IOobject::AUTO_WRITE),
        //         // fvc::grad(D()));
        //         // // newgradD.write();
        //         // Info << "newgradD:" << newgradD[0] << endl;


        //         volSymmTensorField sigma_solution
        //         (
        //             IOobject
        //             (
        //                 "sigma_solution",
        //                 "solutions",
        //                 mesh(),
        //                 IOobject::MUST_READ,
        //                 IOobject::NO_WRITE
        //             ),
        //             mesh()
        //         );

        //         volTensorField F_solution
        //         (
        //             IOobject
        //             (
        //                 "F_solution",
        //                 "solutions",
        //                 mesh(),
        //                 IOobject::MUST_READ,
        //                 IOobject::AUTO_WRITE
        //             ),
        //             mesh()
        //         );

        //         volTensorField Finv_solution
        //         (
        //             IOobject
        //             (
        //                 "Finv_solution",
        //                 "solutions",
        //                 mesh(),
        //                 IOobject::MUST_READ,
        //                 IOobject::AUTO_WRITE
        //             ),
        //             mesh()
        //         );

        //         volScalarField J_solution
        //         (
        //             IOobject
        //             (
        //                 "J_solution",
        //                 "solutions",
        //                 mesh(),
        //                 IOobject::MUST_READ,
        //                 IOobject::AUTO_WRITE
        //             ),
        //             mesh()
        //         );

        //         Finv_ = Finv_solution;
        //         J_ = J_solution;   

        //         volTensorField gradD_solution
        //         (
        //             IOobject
        //             (
        //                 "gradD_solution",
        //                 "solutions",
        //                 mesh(),
        //                 IOobject::MUST_READ,
        //                 IOobject::AUTO_WRITE
        //             ),
        //             mesh()
        //         );

        //         gradD() = gradD_solution;

        //         Info << "Finv_:" << Finv_[0] << endl;
        //         Info << "J_:" << J_[0] << endl;
        //         Info << "D:" << D()[0] << endl;
        //         Info << "gradD(): (solution)" << gradD()[0] << endl;
        //         Info << "sigma_solution:" << sigma_solution[0] << endl;

        //         Info << "sigma_solution: " << sigma_solution << endl;

        //         // GRAD D USED IN CORRECT MATRIX Is FROM PREVIOUS D FIELD NOT SOLUTION D FIELD

        //         // Increment of displacement
        //         DD() = D() - D().oldTime();

        //         // Update gradient of displacement increment
        //         gradDD() = gradD() - gradD().oldTime(); 

        //         Info << "mark1" << endl;

        //         Info << "D before: " <<  D() << endl;

        //         // Momentum equation total displacement total Lagrangian form
        //         fvVectorMatrix DEqn
        //         (
        //             rho()*fvm::d2dt2(D())
        //         == fvm::laplacian(impKf_, D(), "laplacian(DD,D)")
        //         - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
        //         + fvc::div(J_*Finv_ & sigma_solution, "div(sigma)")
        //         + rho()*g()
        //         + stabilisation().stabilisation(DD(), gradDD(), impK_)
        //         ); 

        //         Info << "D after: " <<  D() << endl;


        //         // Info << "matrix 1: " <<  fvc::laplacian(impKf_, D(), "laplacian(DD,D)") << endl;
        //         // Info << "matrix 2: " <<  fvc::div(J_*Finv_ & sigma_solution, "div(sigma)") << endl;
        //         // Info << "matrix 3: " <<  stabilisation().stabilisation(DD(), gradDD(), impK_) << endl;

        //         // Info << DEqn << endl;

        //         // Info <<  D() << endl;
        //         // Info << "matrix 1: " <<  fvc::laplacian(impKf_, D(), "laplacian(DD,D)") << endl;

        //         // Info << DEqn << endl;


        //         // Info << "DD():" << DD()[0] << endl;
        //         // Info << "gradDD():" << gradDD()[0] << endl;


        //         // Info << "mark2" << endl;

        //         // // Under-relax the linear system
        //         // DEqn.relax();

        //         // Enforce any cell displacements
        //         solidModel::setCellDisps(DEqn);

        //         // Hack to avoid expensive copy of residuals
        // #ifdef OPENFOAMESI
        //         const_cast<dictionary&>(mesh().solverPerformanceDict()).clear();
        // #endif
        //         // Finv_.write();
        //         // J_.write();
        //         // sigma().write();         

        //         // Solve the linear system
        //         solverPerfD = DEqn.solve();  

        //         Info << "solverPerfD.initialResidual():" << solverPerfD.initialResidual() << endl;     

            }

            // Turn off prediction switch
            predictionSwitch = false;
        }
        // else
        // {

            // if (iCorr > 600)
            // {
                    
            //     volVectorField D_solution(
            //     IOobject(
            //             "D_solution",
            //             runTime().timeName(),
            //             runTime(),
            //             IOobject::NO_READ,
            //             IOobject::AUTO_WRITE),
            //     D());
            //     D_solution.write();

            //     Info << "D()" << D()[0] << endl;

            //     volSymmTensorField sigma_solution(
            //     IOobject(
            //             "sigma_solution",
            //             runTime().timeName(),
            //             runTime(),
            //             IOobject::NO_READ,
            //             IOobject::AUTO_WRITE),
            //     sigma());
            //     sigma_solution.write();

            //     volTensorField F_solution(
            //     IOobject(
            //             "F_solution",
            //             runTime().timeName(),
            //             runTime(),
            //             IOobject::NO_READ,
            //             IOobject::AUTO_WRITE),
            //     F_);
            //     F_solution.write();

            //     volScalarField J_solution(
            //     IOobject(
            //             "J_solution",
            //             runTime().timeName(),
            //             runTime(),
            //             IOobject::NO_READ,
            //             IOobject::AUTO_WRITE),
            //     J_);
            //     J_solution.write();

            //     volTensorField Finv_solution(
            //     IOobject(
            //             "Finv_solution",
            //             runTime().timeName(),
            //             runTime(),
            //             IOobject::NO_READ,
            //             IOobject::AUTO_WRITE),
            //     Finv_);
            //     Finv_solution.write();

            //     volTensorField gradD_solution(
            //     IOobject(
            //             "gradD_solution",
            //             runTime().timeName(),
            //             runTime(),
            //             IOobject::NO_READ,
            //             IOobject::AUTO_WRITE),
            //     gradD());
            //     gradD_solution.write();
            // }


            // Momentum equation total displacement total Lagrangian form
            fvVectorMatrix DEqn
            (
                rho()*fvm::d2dt2(D())
            == fvm::laplacian(impKf_, D(), "laplacian(DD,D)")
            - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
            + fvc::div(J_*Finv_ & sigma(), "div(sigma)")
            + rho()*g()
            + stabilisation().stabilisation(D(), gradD(), impK_)
            );

            
            // volVectorField residual = 
            //             rho()*fvc::d2dt2(D())
            //             - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
            //             + fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
            //             - fvc::div(J_*Finv_ & sigma(), "div(sigma)")
            //             - rho()*g()
            //             - stabilisation().stabilisation(DD(), gradDD(), impK_);

            // if (Foam::exists(Foam::name(iCorr)))
            // {
            //     Info << "directory exists" << endl;
            // }
            // else
            // {
            //     Info << "Creating directory: " << iCorr << " to store fields" << endl;
            //     Foam::mkDir(Foam::name(iCorr));
            // }

            // // write residual to file
            // volVectorField residual_write
            // (
            //     IOobject
            //     (
            //         "residual",
            //         runTime().timeName(),
            //         "../" + Foam::name(iCorr),
            //         mesh(),
            //         IOobject::NO_READ,
            //         IOobject::AUTO_WRITE
            //     ),
            //     residual
            // );
            // residual_write.write();
            

            // if (iCorr == 669)
            // {
            //     // Info << DEqn << endl;

            //     Info <<  D() << endl;
            //     Info << "matrix 1: " <<  fvc::laplacian(impKf_, D(), "laplacian(DD,D)") << endl;
            //     // Info << "matrix 2: " <<  fvc::div(J_*Finv_ & sigma(), "div(sigma)") << endl;
            //     // Info << "matrix 3: " <<  stabilisation().stabilisation(DD(), gradDD(), impK_) << endl;

            // }

            // Under-relax the linear system
            // DEqn.relax();

            // Enforce any cell displacements
            solidModel::setCellDisps(DEqn);

            // Hack to avoid expensive copy of residuals
    #ifdef OPENFOAMESI
            const_cast<dictionary&>(mesh().solverPerformanceDict()).clear();
    #endif
            // Finv_.write();
            // J_.write();
            // sigma().write();       

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


            // Increment of displacement
            DD() = D() - D().oldTime();

            // Update gradient of displacement
            // calculated using relaxed gradD
            mechanical().grad(D(), gradD());

            // Update gradient of displacement increment
            gradDD() = gradD() - gradD().oldTime();

            // Total deformation gradient
            F_ = I + gradD().T();

            // Inverse of the deformation gradient
            Finv_ = inv(F_);

            // Jacobian of the deformation gradient
            J_ = det(F_);

            // Update the momentum equation inverse diagonal field
            // This may be used by the mechanical law when calculating the
            // hydrostatic pressure
            const volScalarField DEqnA("DEqnA", DEqn.A());

            // Calculate the stress using run-time selectable mechanical law
            mechanical().correct(sigma());


            if (debugFields_)
            {

                if ((iCorr > 18) & (iCorr<30))
                {
                    
                    writeDebugFields(iCorr);
                }

                // if ((iCorr > 128) & (iCorr < 160))
                // {
                    // writeDebugFields(iCorr);
                // }

            }

        // }
    }
    while
    (
        !converged
        (
            iCorr,
#ifdef OPENFOAMESIORFOUNDATION
            mag(solverPerfD.initialResidual()),
            cmptMax(solverPerfD.nIterations()),
#else
            solverPerfD.initialResidual(),
            solverPerfD.nIterations(),
#endif
            D()
        ) && ++iCorr < nCorr()
    );

    // Write final iteration displacement
    writeOutput("convergedD_cells", false);

    // Interpolate cell displacements to vertices
    mechanical().interpolate(D(), pointD());

    // Increment of point displacement
    pointDD() = pointD() - pointD().oldTime();

    // Velocity
    U() = fvc::ddt(D());

#ifdef OPENFOAMESIORFOUNDATION
    SolverPerformance<vector>::debug = 1;
#else
    blockLduMatrix::debug = 1;
#endif

    return true;
}


tmp<vectorField> nonLinGeomTotalLagTotalDispSolidML::tractionBoundarySnGrad
(
    const vectorField& traction,
    const scalarField& pressure,
    const fvPatch& patch
) const
{
    // Patch index
    const label patchID = patch.index();

    // Patch implicit stiffness field
    const scalarField& impK = impK_.boundaryField()[patchID];

    // Patch reciprocal implicit stiffness field
    const scalarField& rImpK = rImpK_.boundaryField()[patchID];

    // Patch gradient
    const tensorField& pGradD = gradD().boundaryField()[patchID];

    // Patch Cauchy stress
    const symmTensorField& pSigma = sigma().boundaryField()[patchID];

    // Patch total deformation gradient inverse
    const tensorField& Finv = Finv_.boundaryField()[patchID];

    // Patch unit normals (initial configuration)
    const vectorField n(patch.nf());

    // Patch unit normals (deformed configuration)
    vectorField nCurrent(Finv.T() & n);
    nCurrent /= mag(nCurrent);

    // Return patch snGrad
    return tmp<vectorField>
    (
        new vectorField
        (
            (
                (traction - nCurrent*pressure)
              - (nCurrent & pSigma)
              + impK*(n & pGradD)
            )*rImpK
        )
    );
}

void nonLinGeomTotalLagTotalDispSolidML::initMLFieldStoring()
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
        // Matrix residual
        if (inputsML_[iterI] == "matrixResidual")
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
            // useCoordinates_ = true;
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

void nonLinGeomTotalLagTotalDispSolidML::storeMLInputFields()
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

        // if (inputsML_[input]=="coordinates")
        // {
        //     //  acces how many dimensions
        //     int dim = inputSizes[input];
            
        //     //  Loop through each dimension creating a scalar field
        //     for (int i = 0; i < dim; i++)
        //     {
        //         scalarField temp = scalarField(D().internalField().size(), 0.0);

        //         forAll(temp, cellI)
        //         {
        //             temp[cellI] = mesh().C()[cellI][i];
        //         }
        //         // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
        //         storedInputFields[index][feature_count + i] = temp;

        //         // Maybe write should be here? 
        //     }            
            
        //     feature_count = feature_count + dim;                  
        // }

        if (inputsML_[input]=="gradD")
        {
            //  access how many dimensions
            int dim = inputSizes[input];

            int j;
       
            //  Loop through each dimension creating a scalar field
            for (int i = 0; i < dim; i++)
            {
                scalarField temp = scalarField(D().internalField().size(), 0.0);

                // For 2D cases do not into the z grads:
                if(dim==6)
                {                
                    if (i == 2)
                        j=3; 
                    else if (i == 3)
                        j=4; 
                    else if(i == 4)
                        j=6; 
                    else if(i == 5)
                        j=7; 
                    else
                        j=i;
                }          

                forAll(temp, cellI)
                {
                    // Info << gradD()[cellI] << endl;
                    temp[cellI] = gradD()[cellI][j];
                }

                // Add to stored list, input is the input category, i.e, D, R etc, i is the dimension. 
                storedInputFields[index][feature_count + i] = temp;
            }      
                
            feature_count = feature_count + dim;                     
        }

        if (inputsML_[input]=="matrixResidual")
        {
            //  acces how many dimensions
            int dim = inputSizes[input];
            
            //  Loop through each dimension creating a scalar field
            for (int i = 0; i < dim; i++)
            {
                scalarField temp = scalarField(D().internalField().size(), 0.0);

                updateMatrixResidual();
                forAll(temp, cellI)
                {
                    temp[cellI] = matrixResidual[cellI][i];
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

void nonLinGeomTotalLagTotalDispSolidML::updateD_ML(int predictionCount, fileName jsonFile, scalar relaxD_ML_)
{

    float start_time = runTime().elapsedCpuTime();

    Info << nl << "Using machine learning to predict D field" << nl << endl;

    Info << "Prediction number: " << predictionCount << endl;

    // Load frugally-deep model
    Info << "Json file location is: " << jsonFile << endl; // move outside of loop

    // // Create the keras model
    auto model = fdeep::load_model(jsonFile);

    Info << "Keras model loaded" << endl;

    float predict_time = 0;

    forAll(D(), cellI)
    {
        int inputVectorSize;
        inputVectorSize = inputSize*machinePredictorIter_;

        if (useCoordinates_) {
            if (predictZ_) {
                inputVectorSize += 3;
            } else {
                inputVectorSize += 2;
            }
        }

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

                // if (cellI==0){
                //     Info << "inputFoam: " << inputFoam[iterI*inputSize + inputF] << endl;
                // }

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

        // // Add coordinate:
        // if (useCoordinates_)
        // {
        //     // int coord_loc = inputVectorSize;
        //     if (predictZ_)
        //     {
        //         inputFoam[inputVectorSize-2] = mesh().C()[cellI].x();
        //         inputFoam[inputVectorSize-1] = mesh().C()[cellI].y();
        //         inputFoam[inputVectorSize] = mesh().C()[cellI].z();

        //         scaledInput[inputVectorSize-2] = 
        //         scale
        //         (
        //             inputFoam[iterI*inputSize + inputVectorSize-2],
        //             kerasScalingMeans_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize-2], 
        //             kerasScalingStds_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize-2]
        //         );

        //         scaledInput[inputVectorSize-1] = 
        //         scale
        //         (
        //             inputFoam[iterI*inputSize + inputVectorSize-1],
        //             kerasScalingMeans_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize-1], 
        //             kerasScalingStds_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize-1]
        //         );

        //         scaledInput[inputVectorSize] = 
        //         scale
        //         (
        //             inputFoam[iterI*inputSize + inputVectorSize],
        //             kerasScalingMeans_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize], 
        //             kerasScalingStds_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize]
        //         );

                
        //     else
        //     {
        //         inputFoam[inputVectorSize-1] = mesh().C()[cellI].x();
        //         inputFoam[inputVectorSize] = mesh().C()[cellI].y();

        //         scaledInput[inputVectorSize-1] = 
        //         scale
        //         (
        //             inputFoam[iterI*inputSize + inputVectorSize-1],
        //             kerasScalingMeans_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize-1], 
        //             kerasScalingStds_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize-1]
        //         );

        //         scaledInput[inputVectorSize] = 
        //         scale
        //         (
        //             inputFoam[iterI*inputSize + inputVectorSize],
        //             kerasScalingMeans_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize], 
        //             kerasScalingStds_[iterI + (predictionCount-1)*(machinePredictorIter_+1)][inputVectorSize]
        //         );
        //     }

        // }

        // Info << "inputFoam: " <<  endl;

        // forAll(inputFoam, i)
        // {
        //     Info << inputFoam[i];
        // }

        float predict_start_time = runTime().elapsedCpuTime();

        const auto prediction = model.predict //_single_output
                                (
                                    {fdeep::tensor(
                                        fdeep::tensor_shape(static_cast<std::size_t>(inputVectorSize)),
                                        scaledInput)});  

        float predict_end_time = runTime().elapsedCpuTime();

        predict_time =  predict_time + predict_end_time - predict_start_time;


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

        // if (cellI==0)
        // {
        //     Info << "result[0] " << result[0] << endl;
        //     Info << "result[1] " << result[1] << endl;
        //     Info << "scaledResult[0] " << scaledResult[0] << endl;
        //     Info << "scaledResult[1] " << scaledResult[1] << endl;
        // }

        D()[cellI].x() = relaxD_ML_*result[0] +  (1-relaxD_ML_)*D()[cellI].x();
        D()[cellI].y() = relaxD_ML_*result[1] +  (1-relaxD_ML_)*D()[cellI].y();

        if (predictZ_)
        {
            D()[cellI].z() = relaxD_ML_*result[2] +  (1-relaxD_ML_)*D()[cellI].z();
        }            
    }

    Info << "Finished prediction" << endl;
    float end_time = runTime().elapsedCpuTime();

    Info << "Time required to predict D: " << end_time - start_time << endl;
    Info << "Time required for model prediction: " << predict_time << endl;



    writeOutput("D_predicted_" + Foam::name(predictionCount), true);



    // writePredictedDField(predictionCount);
}

//  void nonLinGeomTotalLagTotalDispSolidML::BoundaryTractionLoop(fvVectorMatrix DEqn)
void nonLinGeomTotalLagTotalDispSolidML::BoundaryTractionLoop()
{
    Info << "Start of Boundary loop" << endl;
    float start_time = runTime().elapsedCpuTime();


    bool convergedBcTrac = false;

    int BCloopCorr = 0;
    scalar residual = 0;

    // const volScalarField DEqnA("DEqnA", DEqn.A());

    do
    {
        D().storePrevIter();
        Info << "1" << endl;
        // Info << "sigma() before: " <<  sigma()[0] << endl;
        mechanical().correct(sigma());   // Calculates sigma using grad D
        // Info << "sigma() after: " <<  sigma()[0] << endl;
        Info << "2" << endl;
        D().correctBoundaryConditions(); // Applies BCs to boundaries? (changes D)
        Info << "3" << endl;
        mechanical().grad(D(), gradD()); // Calculates new grad D
        Info << "4" << endl;
        // Total deformation gradient
        // Info << "F_ before: " <<  F_[0] << endl;
        F_ = I + gradD().T();
        // Info << "F_ after: " <<  F_[0] << endl;
        Info << "5" << endl;
        // Inverse of the deformation gradient
        // Info << "Finv_ before: " <<  Finv_[0] << endl;
        Finv_ = inv(F_);
        // Info << "Finv_ after: " <<  Finv_[0] << endl;
        Info << "6" << endl;
        // Jacobian of the deformation gradient
        // Info << "J_ before: " <<  J_[0] << endl;
        J_ = det(F_);
        // Info << "J_ after: " <<  J_[0] << endl;


        lduSolverPerformance solverPerfD2;
        fvVectorMatrix DEqn2
        (
            rho()*fvm::d2dt2(D())
        == fvm::laplacian(impKf_, D(), "laplacian(DD,D)")
        - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
        + fvc::div(J_*Finv_ & sigma(), "div(sigma)")
        + rho()*g()
        + stabilisation().stabilisation(D(), gradD(), impK_)
        );

        D().storePrevIter();
        solverPerfD2 = DEqn2.solve();
        residual = solverPerfD2.initialResidual();
        D() = D().prevIter();

        Info << residual << endl;


        // residual =
        //     max(
        //         mag(D() - D().prevIter())
        //         ).value() 
        //         /
        //     (max
        //         (mag(D())).value() + SMALL // SMALL = 1e-15 (defined by OpenFOAM) - To avoid dividing by zero
        //     );
            

        if (residual < tractionBCtol_)
        {
            convergedBcTrac = true;
        }

        BCloopCorr = BCloopCorr + 1;

        Info << "BCloopCorr: " << BCloopCorr << ", residual: " << residual << endl;

    } while (!convergedBcTrac);

    Info << "BC correction loop took " << BCloopCorr << " iterations" << nl;
    
    float end_time = runTime().elapsedCpuTime();
    Info << "Time required for boundary loop: " << end_time - start_time << endl;

    // BCloopCorrFile_() << runTime().timeName() << tab << residual << tab << BCloopCorr << nl;   
}

void nonLinGeomTotalLagTotalDispSolidML::writeOutput(fileName fname, bool volvec)
{
    if (!volvec)
    {
        vectorIOField dataToWrite(
            IOobject(
                // "convergedD_cells",
                fname,
                runTime().timeName(),
                runTime(),
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            D());
        dataToWrite.write();
    }   
    else
    {
        volVectorField D_predicted_write(
                IOobject(
                    fname,
                    runTime().timeName(),
                    runTime(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE),
                D());
        D_predicted_write.write();
    }
}

// Writes individual residual for each direction
void nonLinGeomTotalLagTotalDispSolidML::updateResidualD()
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
scalarField nonLinGeomTotalLagTotalDispSolidML::scalarFieldrelResidual()
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

scalar nonLinGeomTotalLagTotalDispSolidML::residualvf()
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

void nonLinGeomTotalLagTotalDispSolidML::updateD_testConverged()
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

// Writes individual residual for each direction
void nonLinGeomTotalLagTotalDispSolidML::updateMatrixResidual()
{
    matrixResidual =                
            - rho()*fvc::d2dt2(D())
            + fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
            - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
            + fvc::div(J_*Finv_ & sigma(), "div(sigma)")
            // + fvc::div(sigma(), "div(sigma)") 
            + rho()*g()
            + stabilisation().stabilisation(D(), gradD(), impK_);
}

void nonLinGeomTotalLagTotalDispSolidML::writeDebugFields(int iter)
{

    // Matrix residual
    Info << "Calculate matrix Residual" << endl;

    updateMatrixResidual(); 

    volVectorField matrixResidual_write
    (
        IOobject(
            "matrixResidual",
            runTime().timeName(),
            "../" + Foam::name(iter),
            mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
            ),
        D()
    );          

    matrixResidual_write.internalField() = matrixResidual;
    matrixResidual_write.write();

    volVectorField D_write
    (
        IOobject(
            "D",
            runTime().timeName(),
            "../" + Foam::name(iter),
            mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
            ),
        D()
    );          

    D_write.write();

    // volTensorField gradD
    // (
    //     IOobject(
    //         "gradD",
    //         runTime().timeName(),
    //         "../" + Foam::name(iter),
    //         mesh(),
    //         IOobject::NO_READ,
    //         IOobject::AUTO_WRITE
    //         ),
    //     gradD()
    // );          

//  Error

}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solidModels

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
