load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
def repo():
    http_archive(
        name = "eigen3",
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"
        ],
        strip_prefix = 'eigen-3.4.0',
        build_file_content =
    """
EIGEN_FILES = [
    "Eigen/**",
    "Eigen/src/Cholesky/**",
    "Eigen/src/CholmodSupport/**",
    "Eigen/src/Core/**",
    "Eigen/src/Core/*/**",
    "Eigen/src/Core/*/*/**",
    "Eigen/src/Eigenvalues/**",
    "Eigen/src/Geometry/**",
    "Eigen/src/Householder/**",
    "Eigen/src/IterativeLinearSolvers/**",
    "Eigen/src/Jacobi/**",
    "Eigen/src/KLUSupport/**",
    "Eigen/src/LU/**",
    "Eigen/src/MetisSupport/**",
    "Eigen/src/OrderingMethods/**",
    "Eigen/src/PaStiXSupport/**",
    "Eigen/src/PardisoSupport/**",
    "Eigen/src/QR/**",
    "Eigen/src/SPQRSupport/**",
    "Eigen/src/SVD/**",
    "Eigen/src/SparseCholesky/**",
    "Eigen/src/SparseCore/**",
    "Eigen/src/SparseLU/**",
    "Eigen/src/SparseQR/**",
    "Eigen/src/StlSupport/**",
    "Eigen/src/SuperLUSupport/**",
    "Eigen/src/UmfPackSupport/**",
    "Eigen/src/misc/**",
    "Eigen/src/plugins/**",
]

EIGEN_MPL2_HEADER_FILES = glob(
    EIGEN_FILES,
)

cc_library(
    name = "eigen",
    hdrs = EIGEN_MPL2_HEADER_FILES,
    includes = ["."],
    visibility = ["//visibility:public"],
)
    """
    )
