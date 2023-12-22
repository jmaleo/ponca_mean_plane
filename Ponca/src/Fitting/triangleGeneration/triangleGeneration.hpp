
template < class DataPoint, class _WFunctor, typename T>
bool
TriangleGeneration<DataPoint, _WFunctor, T>::computeNeighbors(const DataPoint &evalPoint, const std::vector<VectorType>& _attribNeigs, const std::vector<VectorType>& _normNeigs){
    
    _nb_vt = 0; // Number of valid generated triangles
    _normale = evalPoint.normal();

    if (_method == Method::IndependentGeneration) {
        // Shuffle the neighbors
        std::vector<int> indices(_attribNeigs.size());
        for (int i = 0; i < indices.size(); ++i)
            indices[i] = i;
        std::random_device rd;
        std::mt19937 rg(rd());
        std::shuffle(indices.begin(), indices.end(), rg);

        // Compute the triangles
        _triangles.clear();
        int maxt = std::min(_maxtriangles, (int)_attribNeigs.size()/3);
        auto i = 0; 
        for ( ; _nb_vt < maxt; ++_nb_vt) {
            int i1 = indices[ i++ ];
            int i2 = indices[ i++ ];
            int i3 = indices[ i++ ];
            std::array <VectorType, 3> points = {_attribNeigs[i1], _attribNeigs[i2], _attribNeigs[i3]};
            std::array <VectorType, 3> normals = {_normNeigs[i1], _normNeigs[i2], _normNeigs[i3]};
            _triangles.push_back(Triangle<DataPoint>(points, normals));
        }
    }

    else if (_method == Method::HexagramGeneration) {
        construct_hexa(evalPoint, _attribNeigs, _normNeigs);
    }

    else if (_method == Method::AvgHexagramGeneration) {
        construct_avgHexa(evalPoint, _attribNeigs, _normNeigs);
    }

    if (_method == Method::UniformGeneration || _nb_vt == 0) {
        for (int i = 0; i < _maxtriangles; ++i){
            // rand id
            int i1 = rand() % _attribNeigs.size();
            int i2 = rand() % _attribNeigs.size();
            int i3 = rand() % _attribNeigs.size();
            if (i1 == i2 || i1 == i3 || i2 == i3) continue;
            std::array <VectorType, 3> points = {_attribNeigs[i1], _attribNeigs[i2], _attribNeigs[i3]};
            std::array <VectorType, 3> normals = {_normNeigs[i1], _normNeigs[i2], _normNeigs[i3]};
            _triangles.push_back(Triangle<DataPoint>(points, normals));
            _nb_vt++;
        }
    }
    
    return _nb_vt > 0;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT 
TriangleGeneration<DataPoint, _WFunctor, T>::finalize () {
        
    _A = Scalar(0);
    _H = Scalar(0);
    _G = Scalar(0);

    MatrixType localT = MatrixType::Zero();

    for (int t = 0; t < _nb_vt; ++t) {

        // Simple estimation. 
        Scalar tA = _triangles[t].mu0InterpolatedU();
        if (tA < - CNCEigen_s::epsilon) {
            _A     -= tA;
            _H     += _triangles[t].mu1InterpolatedU(true);
            _G     += _triangles[t].mu2InterpolatedU(true);
            localT += _triangles[t].muXYInterpolatedU(true);
        }
        else if (tA > CNCEigen_s::epsilon) {
            _A     += tA;
            _H     += _triangles[t].mu1InterpolatedU();
            _G     += _triangles[t].mu2InterpolatedU();
            localT += _triangles[t].muXYInterpolatedU();
        }

    } // end for t

    _T11 = localT(0,0);
    _T12 = 0.5 * (localT(0,1) + localT(1,0));
    _T13 = 0.5 * (localT(0,2) + localT(2,0));
    _T22 = localT(1,1);
    _T23 = 0.5 * (localT(1,2) + localT(2,1));
    _T33 = localT(2,2);

    MatrixType _T;

    if (_A != Scalar(0)){
        _T  << _T11, _T12, _T13, 
               _T12, _T22, _T23, 
               _T13, _T23, _T33;
        _T /= _A; 
        _H /= _A;
        _G /= _A;
    }
    else {
        _H = Scalar(0);
        _G = Scalar(0);
    }

    std::tie (k2, k1, v2, v1) = CNCEigen_s::curvaturesFromTensor(_T, 1.0, _normale);

    if (k1 > k2) {
        std::swap(k1, k2);
        std::swap(v1, v2);
    }

    return STABLE;

}

template < class DataPoint, class _WFunctor, typename T>
void
TriangleGeneration<DataPoint, _WFunctor, T>::construct_hexa(const DataPoint &evalPoint, const std::vector<VectorType>& _attribNeigs, const std::vector<VectorType>& _normNeigs) {
    // BIN 
    VectorType c = evalPoint.pos();
    VectorType n = evalPoint.normal();
    VectorType a;
    a.setZero();

    int iSource = 0;
    Scalar avgd = Scalar(0);

    for ( int i = 0 ; i < _attribNeigs.size() ; i++ ) {
        avgd += ( _attribNeigs[ i ] - c ).norm();
        a    += _normNeigs[ i ];
    }

    a /= a.norm();
    n = ( Scalar(1) - _avgnormals ) * n + _avgnormals * a;
    n /= n.norm();
    avgd /= _attribNeigs.size();

    const int m = ( std::fabs( n[0] ) > std::fabs ( n[1] ))
            ? ( ( std::fabs( n[0] ) ) > std::fabs( n[2] ) ? 0 : 2 )
            : ( ( std::fabs( n[1] ) ) > std::fabs( n[2] ) ? 1 : 2 );
    const VectorType e = 
        ( m == 0 ) ? VectorType( 0.0, 1.0, 0.0 ) : 
        ( m == 1 ) ? VectorType( 0.0, 0.0, 1.0 ) :
        VectorType( 1.0, 0.0, 0.0 );

    VectorType u = n.cross( e );
    VectorType v = n.cross( u );
    u /= u.norm();
    v /= v.norm();

    std::array<int, 6> indices = {iSource, iSource, iSource, iSource, iSource, iSource};

    for ( int i = 0 ; i < 6 ; i++ ){
        _distance2 [ i ] = avgd * avgd;
        _targets   [ i ] = avgd * ( u * _cos[ i ] + v * _sin[ i ] );
    }

    for ( int i = 0 ; i < _attribNeigs.size() ; i++ ){
        VectorType p = _attribNeigs[ i ];
        if ( p == c ) continue;
            
        const VectorType w = p - c;
        for ( int j = 0 ; j < 6 ; j++ ){
            const Scalar d2 = ( w - _targets[ j ]).squaredNorm();
            if ( d2 < _distance2[ j ] ){
                indices[ j ] = i;
                _distance2[ j ] = d2;
            }
        }
    }
    std::array <VectorType, 3> t1_points = {_attribNeigs[indices[0]], _attribNeigs[indices[2]], _attribNeigs[indices[4]]};
    std::array <VectorType, 3> t1_normals = {_normNeigs[indices[0]], _normNeigs[indices[2]], _normNeigs[indices[4]]};

    std::array <VectorType, 3> t2_points = {_attribNeigs[indices[1]], _attribNeigs[indices[3]], _attribNeigs[indices[5]]};
    std::array <VectorType, 3> t2_normals = {_normNeigs[indices[1]], _normNeigs[indices[3]], _normNeigs[indices[5]]};

    _triangles.push_back(Triangle<DataPoint>(t1_points, t1_normals));
    _triangles.push_back(Triangle<DataPoint>(t2_points, t2_normals));

    _nb_vt = 2;
}

template < class DataPoint, class _WFunctor, typename T>
void
TriangleGeneration<DataPoint, _WFunctor, T>::construct_avgHexa(const DataPoint &evalPoint, const std::vector<VectorType>& _attribNeigs, const std::vector<VectorType>& _normNeigs) {
    
    const VectorType c = evalPoint.pos();
    const VectorType ni = evalPoint.normal();
    VectorType n = ni;
    Scalar avgd = Scalar(0);
    VectorType a;
    a.setZero();

    std::array< VectorType,6 > array_avg_normals;
    std::array< VectorType,6 > array_avg_points;  
    std::array< size_t, 6 >    array_nb;

    for ( int j = 0 ; j < _attribNeigs.size() ; j++ ) {
        avgd += ( _attribNeigs[ j ] - c ).norm();
        a    += _normNeigs[ j ];
    }

    a    /= a.norm();
    n     = ( 1.0 - _avgnormals ) * n + _avgnormals * a;
    n    /= n.norm();
    avgd /= _attribNeigs.size();


    const int m = ( std::fabs( n[ 0 ] ) > std::fabs ( n[ 1 ] ) )
            ? ( ( std::fabs( n[ 0 ] ) ) > std::fabs( n[ 2 ] ) ? 0 : 2 )
            : ( ( std::fabs( n[ 1 ] ) ) > std::fabs( n[ 2 ] ) ? 1 : 2 );
    const VectorType e = 
        ( m == 0 ) ? VectorType( 0.0, 1.0, 0.0 ) : 
        ( m == 1 ) ? VectorType( 0.0, 0.0, 1.0 ) :
        VectorType( 1.0, 0.0, 0.0 );

    VectorType u = n.cross( e );
    VectorType v = n.cross( u );
    u /= u.norm();
    v /= v.norm();
    const VectorType zero = VectorType::Zero();

    for (int i = 0 ; i < 6 ; i++ ){
        _targets   [ i ] = avgd * ( u * _cos[ i ] + v * _sin[ i ] );
        array_avg_normals[ i ] = zero;
        array_avg_points[ i ] = zero;
        array_nb[i] = 0;
    }

    for (int i = 0 ; i < _attribNeigs.size() ; i++ ){
        VectorType w = _attribNeigs[ i ] - c;
        int best_k = 0;
        double best_d2 = ( w - _targets[ 0 ] ).squaredNorm();
        for (auto k = 1 ; k < 6 ; k++ ){
            const double d2 = ( w - _targets[ k ] ).squaredNorm();
            if ( d2 < best_d2 ){
                best_k  = k;
                best_d2 = d2;
            }
        }
        array_avg_normals[ best_k ] += _normNeigs[ i ];
        array_avg_points[ best_k ] += _attribNeigs[ i ];
        array_nb[ best_k ] += 1;
    }

    for (int i = 0 ; i < 6 ; i++ ){
        if ( array_nb[ i ] == 0 ) {
            array_avg_normals[ i ] = evalPoint.normal();
            array_avg_points [ i ] = evalPoint.pos();
        }
        else {
            array_avg_normals[ i ] /= array_avg_normals[ i ].norm();
            array_avg_points [ i ] /= array_nb[ i ];
        }
    }


    std::array <VectorType, 3> t1_points = { array_avg_points[0], array_avg_points[2], array_avg_points[4] };
    std::array <VectorType, 3> t1_normals = { array_avg_normals[0], array_avg_normals[2], array_avg_normals[4] };

    std::array <VectorType, 3> t2_points = { array_avg_points[1], array_avg_points[3], array_avg_points[5] };
    std::array <VectorType, 3> t2_normals = { array_avg_normals[1], array_avg_normals[3], array_avg_normals[5] };

    _triangles.push_back(Triangle<DataPoint>(t1_points, t1_normals));
    _triangles.push_back(Triangle<DataPoint>(t2_points, t2_normals));

    _nb_vt = 2;
}
