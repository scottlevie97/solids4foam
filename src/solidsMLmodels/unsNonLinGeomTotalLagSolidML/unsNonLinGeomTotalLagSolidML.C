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

#include "unsNonLinGeomTotalLagSolidML.H"
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

defineTypeNameAndDebug(unsNonLinGeomTotalLagSolidML, 0);
addToRunTimeSelectionTable
(
    solidModel, unsNonLinGeomTotalLagSolidML, dictionary
);


// * * * * * * * * * * *  Private Member Functions * * * * * * * * * * * * * //


scalar unsNonLinGeomTotalLagSolidML::residual(const volVectorField& D) const
{
    return
        gMax
        (
#ifdef OPENFOAMESIORFOUNDATION
            DimensionedField<double, volMesh>
#endif
            (
                mag(D.internalField() - D.prevIter().internalField())
               /max
                (
                    gMax
                    (
#ifdef OPENFOAMESIORFOUNDATION
                        DimensionedField<double, volMesh>
#endif
                        (
                            mag
                            (
                                D.internalField() - D.oldTime().internalField()
                            )
                        )
                    ),
                    SMALL
                )
            )
        );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

unsNonLinGeomTotalLagSolidML::unsNonLinGeomTotalLagSolidML
(
    Time& runTime,
    const word& region
)
:
    solidModel(typeName, runTime, region),
    sigmaf_
    (
        IOobject
        (
            "sigmaf",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh(),
        dimensionedSymmTensor("zero", dimForce/dimArea, symmTensor::zero)
    ),
    gradDf_
    (
        IOobject
        (
            "grad(" + D().name() + ")f",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        mesh(),
        dimensionedTensor("0", dimless, tensor::zero)
    ),
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
    Ff_
    (
        IOobject
        (
            "Ff",
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
    Finvf_
    (
        IOobject
        (
            "Finvf",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        inv(Ff_)
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
    Jf_
    (
        IOobject
        (
            "Jf",
            runTime.timeName(),
            mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        det(Ff_)
    ),
    impK_(mechanical().impK()),
    impKf_(mechanical().impKf()),
    rImpK_(1.0/impK_),
    nonLinear_(solidModelDict().lookupOrDefault<Switch>("nonLinear", true)),
    debug_(solidModelDict().lookupOrDefault<Switch>("debug", false)),
    K_
    (
        solidModelDict().lookupOrDefault<dimensionedScalar>
        (
            "K",
            dimensionedScalar("K", dimless/dimTime, 0)
        )
    ),
    relativeTol_
    (
        solidModelDict().lookupOrDefault<scalar>
        (
            "solutionTolerance",
            solutionTol()
        )
    ),
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
    BCloopCorrFile_()
{
    DisRequired();

    // For consistent restarts, we will update the relative kinematic fields
    D().correctBoundaryConditions();
    if (restart())
    {
        mechanical().interpolate(D(), pointD(), false);
        mechanical().grad(D(), pointD(), gradD());
        mechanical().grad(D(), pointD(), gradDf_);
        gradDD() = gradD() - gradD().oldTime();
        Ff_ = I + gradDf_.T();
        Finvf_ = inv(Ff_);
        Jf_ = det(Ff_);

        gradD().storeOldTime();

        // Let the mechanical law know
        mechanical().setRestart();
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


bool unsNonLinGeomTotalLagSolidML::evolve()
{
    Info<< "Evolving solid solver" << endl;

    int iCorr = 0;
    scalar initialResidual = 0;
#ifdef OPENFOAMESIORFOUNDATION
    SolverPerformance<vector> solverPerfD;
    SolverPerformance<vector>::debug = 0;
#else
    lduSolverPerformance solverPerfD;
    blockLduMatrix::debug = 0;
#endif
    scalar res = 1.0;
    scalar maxRes = 0;
    scalar curConvergenceTolerance = solutionTol();

    // Reset enforceLinear switch
    enforceLinear() = false;

    int startIter = 0;
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


    do
    {
#ifdef OPENFOAMESIORFOUNDATION
        if (SolverPerformance<vector>::debug)
#else
        if (blockLduMatrix::debug)
#endif
        {
            Info<< "Time: " << runTime().timeName()
                << ", outer iteration: " << iCorr << endl;
        }

        // The first residual when iCorr=0 is 0
        // Using this method will never use the resiudal when iCorr=0
        // This is why the results will be different to linGeomTotatlDispSolidML/linGeomTotatlDispSolidML.C
        if (predictionCount < noPredictions_ )
        {
            if ((res < predictionResiduals_[predictionCount]) && (iCorr > 0))
            {
                Info << "At iteration: " << iCorr << " residual: " << res
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
                Info << "Adding D from iteration " << iCorr << " for residual " << res << endl;
                index = iCorr - startIter;
                Info << "Index: " << index << endl;

                storeMLInputFields();                        
            }
        }

        // Store previous iteration to allow under-relaxation and residual
        // calculation
        D().storePrevIter();

        if (iCorr == machinePredictorIter_ + startIter && predictionSwitch)
        {
            if (machineLearning_)
            {
                Info << "iCorr: " << iCorr << endl;
                Info << "res: " << res << endl;
                Info << "predictionCount: " << predictionCount << endl;

                // Predict displacement
                updateD_ML(predictionCount, jsonFiles_[predictionCount - 1], relaxD_ML_);
                // Boundary traction loop
                BoundaryTractionLoop();        
            }

            // if (testConverged_)
            // {
            //     updateD_testConverged();
               
            //     //  Note: this stores D.prevIter
            //     BoundaryTractionLoop();
            // }

            // Turn off prediction switch
            predictionSwitch = false;
        }

        // Construct momentum equation in total Lagrangian form where gradients
        // are calculated directly at the faces
        fvVectorMatrix DEqn
        (
            rho()*fvm::d2dt2(D())
         == fvm::laplacian(impKf_, D(), "laplacian(DD,D)")
          - fvc::laplacian(impKf_, D(), "laplacian(DD,D)")
          + fvc::div((Jf_*Finvf_.T() & mesh().Sf()) & sigmaf_)
          + rho()*g()
        );

        // Add damping
        if (K_.value() > SMALL)
        {
            DEqn += K_*rho()*fvm::ddt(D());
        }

        // Enforce linear to improve convergence
        if (enforceLinear())
        {
            // Replace nonlinear terms with linear
            // Note: the mechanical law could still be nonlinear
            DEqn +=
                fvc::div((Jf_*Finvf_.T() & mesh().Sf()) & sigmaf_)
              - fvc::div(mesh().Sf() & sigmaf_);
        }

        // Under-relax the linear system
        DEqn.relax();

        // Enforce any cell displacements
        solidModel::setCellDisps(DEqn);

        // Hack to avoid expensive copy of residuals
#ifdef OPENFOAMESI
        const_cast<dictionary&>(mesh().solverPerformanceDict()).clear();
#endif

        // Solve the system
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

        if (iCorr == 0)
        {
            initialResidual = mag(solverPerfD.initialResidual());
        }

        // Interpolate D to pointD
        mechanical().interpolate(D(), pointD(), false);

        // Update gradient of displacement
        mechanical().grad(D(), pointD(), gradD());
        mechanical().grad(D(), pointD(), gradDf_);

        // Update gradient of displacement increment
        gradDD() = gradD() - gradD().oldTime();

        // Total deformation gradient
        Ff_ = I + gradDf_.T();

        // Inverse of the deformation gradient
        Finvf_ = inv(Ff_);

        // Jacobian of the deformation gradient
        Jf_ = det(Ff_);

        // Check if outer loops are diverging
        if (nonLinear_ && !enforceLinear())
        {
            checkEnforceLinear(Jf_);
        }

        // Calculate the stress using run-time selectable mechanical law
        mechanical().correct(sigmaf_);

        // Calculate relative momentum residual
        res = residual(D());

        if (res > maxRes)
        {
            maxRes = res;
        }

        curConvergenceTolerance = maxRes*relativeTol_;

        if (curConvergenceTolerance < solutionTol())
        {
            curConvergenceTolerance = solutionTol();
        }

        if
        (
#ifdef OPENFOAMESIORFOUNDATION
            SolverPerformance<vector>::debug
#else
            blockLduMatrix::debug
#endif
         || (iCorr % infoFrequency()) == 0
         || res < curConvergenceTolerance
         || maxIterReached() == nCorr()
        )
        {
            Info<< "Corr " << iCorr << ", Time " << runTime().elapsedCpuTime()  << ", relative residual = " << res << endl;

            // write values here
        }

        if (maxIterReached() == nCorr())
        {
            maxIterReached()++;
        }

        // Force at least one iteration
	if (iCorr == 0)
        {
            res = 1.0;
        }
    }
    // while (res > curConvergenceTolerance && ++iCorr < nCorr());
    while (
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
    writeOutput();

    // Velocity
    U() = fvc::ddt(D());

    // Total deformation gradient
    F_ = I + gradD().T();

    // Inverse of the deformation gradient
    Finv_ = inv(F_);

    // Jacobian of the deformation gradient
    J_ = det(F_);

    // Calculate the stress using run-time selectable mechanical law
    mechanical().correct(sigma());

    // Increment of displacement
    DD() = D() - D().oldTime();

    // Increment of point displacement
    pointDD() = pointD() - pointD().oldTime();

    // Print summary of residuals
    Info<< solverPerfD.solverName() << ": Solving for " << D().name()
        << ", Initial residual = " << initialResidual
        << ", Final residual = " << solverPerfD.initialResidual()
        << ", No outer iterations = " << iCorr << nl
        << " Max relative residual = " << maxRes
        << ", Relative residual = " << res
        << ", enforceLinear = " << enforceLinear() << endl;

#ifdef OPENFOAMESIORFOUNDATION
    SolverPerformance<vector>::debug = 1;
#else
    blockLduMatrix::debug = 1;
#endif

    if (nonLinear_ && enforceLinear())
    {
        return false;
    }

    return true;
}


tmp<vectorField> unsNonLinGeomTotalLagSolidML::tractionBoundarySnGrad
(
    const vectorField& traction,
    const scalarField& pressure,
    const fvPatch& patch
) const
{
    // Patch index
    const label patchID = patch.index();

    // Patch mechanical property
    const scalarField& impK = impKf_.boundaryField()[patchID];

    // Patch reciprocal implicit stiffness field
    const scalarField& rImpK = rImpK_.boundaryField()[patchID];

    // Patch gradient
    const tensorField& gradD = gradDf_.boundaryField()[patchID];

    // Patch stress
    const symmTensorField& sigma = sigmaf_.boundaryField()[patchID];

    // Patch unit normals (initial configuration)
    const vectorField n(patch.nf());

    if (enforceLinear())
    {
        // Return patch snGrad
        return tmp<vectorField>
        (
            new vectorField
            (
                (
                    (traction - n*pressure)
                  - (n & sigma)
                  + (n & (impK*gradD))
                )*rImpK
            )
        );
    }
    else
    {
        // Patch total deformation gradient inverse
        const tensorField& Finv = Finvf_.boundaryField()[patchID];

        // Patch total Jacobian
        const scalarField& J = Jf_.boundaryField()[patchID];

        // Patch unit normals (deformed configuration)
        const vectorField nCurrent(J*Finv.T() & n);

        // Return patch snGrad
        return tmp<vectorField>
        (
            new vectorField
            (
                (
                    (traction - nCurrent*pressure)
                  - (nCurrent & sigma)
                  + (n & (impK*gradD))
                )*rImpK
            )
        );
    }
}

void unsNonLinGeomTotalLagSolidML::initMLFieldStoring()
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

void unsNonLinGeomTotalLagSolidML::storeMLInputFields()
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

void unsNonLinGeomTotalLagSolidML::updateD_ML(int predictionCount, fileName jsonFile, scalar relaxD_ML_)
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

            if (cellI==0){

                Info << "Iteration: " << iterI << endl;

            }

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

 void unsNonLinGeomTotalLagSolidML::BoundaryTractionLoop()
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

    } while (!convergedBcTrac);

    Info << "BC correction loop took " << BCloopCorr << " iterations" << nl;

    // BCloopCorrFile_() << runTime().timeName() << tab << residual << tab << BCloopCorr << nl;   
}

void unsNonLinGeomTotalLagSolidML::writeOutput()
{
    vectorIOField dataToWrite(
        IOobject(
            "convergedD_cells",
            runTime().timeName(),
            runTime(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        D());
    dataToWrite.write();
}

// Writes individual residual for each direction
void unsNonLinGeomTotalLagSolidML::updateResidualD()
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
scalarField unsNonLinGeomTotalLagSolidML::scalarFieldrelResidual()
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

void unsNonLinGeomTotalLagSolidML::writePredictedDField(int predictionCount)
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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solidModels

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
