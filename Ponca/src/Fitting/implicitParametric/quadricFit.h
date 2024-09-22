#pragma once


#include <Eigen/Core>
#include <Ponca/Fitting>

namespace Ponca
{

/*!
    \brief Algebraic quadric primitive

    \see Quadric
*/
template < class DataPoint, class _WFunctor, typename T >
class QuadricFitImpl : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        check = Base::PROVIDES_ALGEBRAIC_QUADRIC
    };

public:
    PONCA_EXPLICIT_CAST_OPERATORS(QuadricFitImpl, quadricFitImpl)

public:

    using VectorA = typename Base::VectorA;
    using MatrixA = typename Base::MatrixA;

    using Matrix32 = Eigen::Matrix<Scalar, 3, 2>;

    using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
    using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;

// results
public:

    MatrixA m_matA;
    int m_minId;
    // EigenSolver of m_matA
    Eigen::EigenSolver<MatrixA> m_solver;

public:
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    // PONCA_MULTIARCH inline const Solver& solver() const { return m_solver; }

    PONCA_MULTIARCH inline FIT_RESULT constructTensor ();

    PONCA_MULTIARCH inline FIT_RESULT constructTensor2 ();

protected:

    PONCA_MULTIARCH Matrix32 tangentPlane(const VectorType& n);

}; //class Quadric

template < class DataPoint, class _WFunctor, typename T>
using QuadricFit = 
    QuadricFitImpl<DataPoint, _WFunctor, 
        Quadric<DataPoint, _WFunctor,
            Ponca::LocalFrame<DataPoint, _WFunctor,
                Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>;


#include "quadricFit.hpp"
} //namespace Ponca
