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

Class
    nonLinGeomTotalLagTotalDispSolidML

Description
    Mathematical model for finite strains and rotations, where a total
    Lagrangian and total displacement approach is employed.

    The stress is calculated by the run-time selectable mechanical law.

    More details in:
    P. Cardiff, Z. Tukovic, P. De Jaeger, M. Clancy and A. Ivankovic. A
    Lagrangian cell-centred finite volume method for metal forming simulation,
    doi=10.1002/nme.5345.

Author
    Philip Cardiff, UCD.  All rights reserved.

SourceFiles
    nonLinGeomTotalLagTotalDispSolidML.C

\*---------------------------------------------------------------------------*/

#ifndef nonLinGeomTotalLagTotalDispSolidML_H
#define nonLinGeomTotalLagTotalDispSolidML_H

#include "solidModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace solidModels
{

/*---------------------------------------------------------------------------*\
                        Class nonLinGeomTotalLagTotalDispSolidML Declaration
\*---------------------------------------------------------------------------*/

class nonLinGeomTotalLagTotalDispSolidML
:
    public solidModel
{
    // Private data

        //- Total deformation gradient
        volTensorField F_;

        //- Inverse of the total deformation gradient
        volTensorField Finv_;

        //- Jacobian of total deformation gradient
        volScalarField J_;

        //- Implicit stiffness; coefficient of the Laplacian term
        //  The value of this term only affects convergence and not the answer
        //  In the case of Hookean elasticity, (2*mu + lambda) is the optimum
        const volScalarField impK_;

        //- Implicit stiffness surface field
        const surfaceScalarField impKf_;

        //- Reciprocal of impK
        //  We divide by impK every time tractionBoundarySnGrad is called so we
        //  will store the reciprocal for effficiency
        const volScalarField rImpK_;

        //- Predict new time-step fields using the velocity field
        const Switch predictor_;


        //- Iteration at which the machine learning predictor is applied
        const int machinePredictorIter_;

        // - Flag to switch to machine learning prediction
        const Switch machineLearning_;

        //- JSON file name
        List<fileName> jsonFiles_;

        // ML relax
        scalar relaxD_ML_;

        //- Number of input features
        int inputSize;

        // ML input features
        List<fileName> inputsML_;

        //- Predict X, Y and Z fields. This should be switched off for 2D cases.
        const Switch predictZ_;

        // //List of scaling means and std deviations (input data | output data)
        List<scalarField> kerasScalingMeans_;
        List<scalarField> kerasScalingStds_;

        // The residuals where the predicitons will begin to collect data
        List<scalar> predictionResiduals_;
        
        List<scalar> inputSizes;

        //- Stored input fields
        List<PtrList<scalarField>> storedInputFields;

        //  Relative residual field
        vectorField residualD;

        //- Count number of predictions that have been made
        int predictionCount;

        // The amount of predictions
        scalar noPredictions_;

        // Prediction index
        int index;

        // Write ML feature fields
        const Switch  writeFields_;

        // Boundary traction loop
        scalar tractionBCtol_;
        
        //-  BC Correction Loop File
        autoPtr<OFstream> BCloopCorrFile_;

        // - Flag to switch to test converged values
        const Switch testConverged_;

        //- Converged case path for test convergence
        fileName convergedCase_;

        // - Offset iteration
        int offsetIter_;

        //  Relative residual field
        vectorField matrixResidual;

        const Switch debugFields_;

        //- Use cell centre coordinates as input feature
        const Switch useCoordinates_;


    // Private Member Functions

        //- Predict the fields for the next time-step based on the
        //  previous time-steps
        void predict();

        //- Disallow default bitwise copy construct
        nonLinGeomTotalLagTotalDispSolidML
        (
            const nonLinGeomTotalLagTotalDispSolidML&
        );

        //- Disallow default bitwise assignment
        void operator=(const nonLinGeomTotalLagTotalDispSolidML&);

        // Scaling functions
        double scale(double x, double u, double s) // change double to scalar
        {
            return (x - u) / (s); // + SMALL);
        }

        double invScale(double z, double u, double s)
        {
            return z * s + u;
        }


protected:

    // Protected member functions

        //- Return nonlinear geometry enumerator
        virtual nonLinearGeometry::nonLinearType nonLinGeom() const
        {
            return nonLinearGeometry::TOTAL_LAGRANGIAN;
        }


public:

    //- Runtime type information
    TypeName("nonLinearGeometryTotalLagrangianTotalDisplacementML");

    // Constructors

        //- Construct from components
        nonLinGeomTotalLagTotalDispSolidML
        (
            Time& runTime,
            const word& region = dynamicFvMesh::defaultRegion
        );


    // Destructor

        virtual ~nonLinGeomTotalLagTotalDispSolidML()
        {}


    // Member Functions

        // Access

            //- Each solidModel must indicate whether D or DD is the primary
            //  solution variable
            virtual volVectorField& solutionD()
            {
                // This model solves for D
                return D();
            }

        // Edit

            //- Evolve the solid solver and solve the mathematical model
            virtual bool evolve();

            //- Traction boundary surface normal gradient
            virtual tmp<vectorField> tractionBoundarySnGrad
            (
                const vectorField& traction,
                const scalarField& pressure,
                const fvPatch& patch
            ) const;

            //- Initialise ML feature storing
            virtual void initMLFieldStoring();

            //- Store ML input features
            virtual void storeMLInputFields();

            //- Set D using machine learning predictor
            virtual void updateD_ML(int predictionCount, fileName jsonFile, scalar relaxD_ML);

            //- Boundary Condition Traction Loop
            // virtual void BoundaryTractionLoop(fvVectorMatrix DEqn);
            virtual void BoundaryTractionLoop();


            //- Write converged solution
            virtual void writeOutput(fileName fname, bool volvec);

            //- Calculate relative residual field
            virtual void updateResidualD();

            scalarField scalarFieldrelResidual();

            scalar residualvf();

            virtual void updateD_testConverged();

            //  - Calculate matrix residual field
            virtual void updateMatrixResidual();
        
            // Write debug fields into time directories
            virtual void writeDebugFields(int iter);

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solidModel

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
