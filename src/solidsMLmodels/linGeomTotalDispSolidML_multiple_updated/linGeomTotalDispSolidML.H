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

Class
    linGeomTotalDispSolidML

Description
    Mathematical model where linear geometry is assumed i.e. small strains and
    small rotations are assumed, and the total displacement is the primary
    unknown. Machine learning models are used predict converged values.

    The stress is calculated by the run-time selectable mechanical law.

Author
    Philip Cardiff, Scott Levie UCD.  All rights reserved.

SourceFiles
    linGeomTotalDispSolidML.C

\*---------------------------------------------------------------------------*/

#ifndef linGeomTotalDispSolidML_H
#define linGeomTotalDispSolidML_H

#include "solidModel.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "pointFields.H"
#include "uniformDimensionedFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    namespace solidModels
    {

        /*---------------------------------------------------------------------------*\
                                Class linGeomTotalDispSolidML Declaration
        \*---------------------------------------------------------------------------*/

        class linGeomTotalDispSolidML
            : public solidModel
        {
            // Private data

            //- Implicit stiffness; coefficient of the Laplacian term
            //  The value of this term only affects convergence and not the answer
            //  In the case of Hookean elasticity, (2*mu + lambda) is the optimum
            const volScalarField impK_;

            //- Implicit stiffness surface field
            surfaceScalarField impKf_;

            //- Reciprocal of impK
            //  We divide by impK every time tractionBoundarySnGrad is called so we
            //  will store the reciprocal for effficiency
            const volScalarField rImpK_;

            //- Predict new time-step fields using the velocity field
            const Switch predictor_;

            //- Write file with individual residuals for each direction
            autoPtr<OFstream> indivResidualFilePtr_;
            const Switch writeIndivResidualFile_;

            // - Flag to switch to machine learning prediction
            const Switch machineLearning_;

            // - Flag to use residual field as input
            const Switch useResiduals_;

            //- Iteration at which the machine learning predictor is applied
            const int machinePredictorIter_;

            //- JSON file name
            List<fileName> jsonFiles_;

            //- Converged case path for test convergence
            fileName convergedCase_;

            //- List of cell D fields for previous
            //  iterations. These are used by the machine learning predictor
            PtrList<vectorField> prevCellD_;

            PtrList<vectorField> prevCellR_;


            // - Flag to switch to test converged values
            const Switch testConverged_;

            //-  BC Correction Loop File
            autoPtr<OFstream> BCloopCorrFile_;

            const int writeCellDisplacement_;

            autoPtr<OFstream> cellDisplacementFile_;

            List<scalar> cellList_;

            //- Write displacement fields for each iteration
            const Switch writeDisplacementField_;

            //- Write displacements up until this iteration
            const int writeDisplacementLimit_;

            //- Use cell centre coordinates as input feature
            const Switch useCoordinates_;

            //- Predict X, Y and Z fields. This should be switched off for 2D cases.
            const Switch predictZ_;

            // The amount of predictions
            scalar noPredictions_;

            // ML relax
            scalar relaxD_ML_;

            const Switch debugFields_;

            //- Hooke's law second Lame parameter, aka shear modulus
            surfaceScalarField muf_;

            //- Hooke's law first Lame parameter
            surfaceScalarField lambdaf_;

            // //List of scaling means and std deviations (input data | output data)
            List<vector> kerasScalingMeans_;
            List<vector> kerasScalingStds_;

            // The residuals where the predicitons will begin to collect data
            List<scalar> predictionResiduals_;

            //  Relative residual field
            vectorField residualD;


            // Private Member Functions

            //- Predict the fields for the next time-step based on the
            //  previous time-steps
            void predict();

            //- Disallow default bitwise copy construct
            linGeomTotalDispSolidML(const linGeomTotalDispSolidML &);

            //- Disallow default bitwise assignment
            void operator=(const linGeomTotalDispSolidML &);

            // Scaling function

            double scale(double x, double u, double s) // change double to scalar
            {
                return (x - u) / s;
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
                return nonLinearGeometry::LINEAR_GEOMETRY;
            }

        public:
            //- Runtime type information
            TypeName("linearGeometryTotalDisplacementML");

            // Constructors

            //- Construct from components
            linGeomTotalDispSolidML(
                Time &runTime,
                const word &region = dynamicFvMesh::defaultRegion);

            // Destructor

            virtual ~linGeomTotalDispSolidML()
            {
            }

            // Member Functions

            // Access

            //- Each solidModel must indicate whether D or DD is the primary
            //  solution variable
            virtual volVectorField &solutionD()
            {
                // This model solves for D
                return D();
            }

            // Edit

            //- Evolve the solid solver and solve the mathematical model
            virtual bool evolve();

            //- Set D using machine learning predictor
            virtual void updateD_ML(int predictionCount, fileName jsonFile, scalar relaxD_ML);

            // - Set D using machine learning predictor
            virtual void updateD_testConverged();

            // - Set Sigma using machine learning predictor
            virtual void updateSigma_testConverged();

            // - Boundary Condition Traction Loop
            virtual void BoundaryTractionLoop();

            // - Write Displacement for each iteration
            virtual void writeDisplacementIteration(int predictionCount, int iter, bool converged);

            // - Write Residual for each iteration
            virtual void writeResidualIteration(int predictionCount, int iter, bool converged);
        
            // Write debug fields into time directories
            virtual void writeDebugFields(int iter);

            // - Write Displacement for prediction iteration
            virtual void writePredictedDField(int predictionCount);

            // - Write Displacement for prediction iteration
            virtual void writeResidualFile(int iteration);

            //  - Calculate relative residual field
            virtual void updateResidualD();

            virtual void writeCellDisplacementList(int iteration);

            // // - write residual field
            // virtual void writeResidualField(int iCorr);

            //- Traction boundary surface normal gradient
            virtual tmp<vectorField> tractionBoundarySnGrad(
                const vectorField &traction,
                const scalarField &pressure,
                const fvPatch &patch) const;

            scalar residualvf();


        };

        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    } // End namespace solidModel

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ********************************j***************************************** //
