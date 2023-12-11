template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{
    VectorType x = Base::worldToLocalFrame(_q);
    return eval_quadratic_function(*(x.data() +1 ), *(x.data() + 2));
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    if (m_isCylinder) {
        VectorType x = Base::worldToLocalFrame(_q);
        VectorType grad = Base::primitiveGradient(_q);
        return Base::project(_q) + ( eval_quadratic_function(*(x.data()+1), *(x.data()+2)) * grad);
    }
    else {
        const VectorType grad = primitiveGradient(_q);

        // Grab the normal (primitiv gradient)
        VectorType u = Base::getFrameU();
        VectorType v = Base::getFrameV();
        VectorType n = primitiveGradient();

        MatrixType B;
        B << n, u, v;

        Matrix2 dN_2D = (2 * m_a * m_uq);
        MatrixType dN = MatrixType::Zero();
        dN.block(1,1,2,2) = m_correctOrientation * dN_2D;
        // put 1 on the first row and column
        dN(0,0) = 1; 

        const Scalar a = grad.transpose() * dN * grad;
        const Scalar b = grad.squaredNorm();
        const Scalar c = potential(_q);

        // solve a t^2 + b t + c = 0
        Scalar t = 0;

        if(abs(a) < epsilon)
        {
            if(abs(b) < epsilon)
            {
                t = 0;
            }
            else
            {
                t = - c / b;
            }
        }
        else
        {
            const Scalar delta = b*b - 4*a*c;

            if(delta >= 0)
            {
                t = (- b + sqrt(delta)) / (2 * a);
            }
            else
            {
                t = 0; // no solution so no projection
            }
        }

        return _q + t * grad;
    }
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType& _q ) const
{
    // Convexe = m_a >= 0    Concave = m_a <= 0

    VectorType proj = Base::worldToLocalFrame(_q);
    Vector2 temp {proj(1),  proj(2)};
    Vector2 df = m_ul + 2 * m_a * m_uq * temp;
    df *= m_correctOrientation;
    VectorType local_gradient { 1, df(0) , df(1) };

    VectorType world_gradient = Base::template localFrameToWorld<true>(local_gradient);

    return world_gradient;
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::MatrixType
ParabolicCylinder<DataPoint, _WFunctor, T>::dNormal() const
{ 
    // Grab the normal (primitiv gradient)
    VectorType u = Base::getFrameU();
    VectorType v = Base::getFrameV();
    VectorType n = primitiveGradient();

    MatrixType B;
    B << n, u, v;

    Matrix2 dN_2D = (2 * m_a * m_uq);
    MatrixType dN = MatrixType::Zero();
    dN.block(1,1,2,2) = dN_2D;
    // put 1 on the first row and column
    dN(0,0) = 1; 

    VectorType ul {1, -m_ul(0), -m_ul(1)};
    
    return ( B * dN * B.transpose() ) / Base::template localFrameToWorld<true>(ul).norm(); 

}