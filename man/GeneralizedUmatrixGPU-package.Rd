\name{GeneralizedUmatrixGPU-package}
\alias{GeneralizedUmatrixGPU-package}
\docType{package}
\title{
\packageTitle{GeneralizedUmatrixGPU}
}
\description{
\packageDescription{GeneralizedUmatrixGPU}
}
\details{
For a brief introduction to \pkg{GeneralizedUmatrixGPU} please see the vignette of the CRAN package \href{https://CRAN.R-project.org/package=GeneralizedUmatrix}{GeneralizedUmatrix}.

For further details regarding the generalized Umatrix see [Thrun, 2018], chapter 4-5, or [Thrun/Ultsch, 2020].

If you want to verifiy your clustering result externally, you can use \code{Heatmap} or \code{SilhouettePlot} of the CRAN package \code{DataVisualizations}.

\packageIndices{GeneralizedUmatrixGPU}
}
\author{Quirin Stier}
\references{
[Thrun/Ultsch, 2020]  Thrun, M. C., & Ultsch, A.: Uncovering High-Dimensional Structures of Projections from Dimensionality Reduction Methods, MethodsX, Vol. 7, pp. 101093, DOI \doi{10.1016/j.mex.2020.101093}, 2020.

[Thrun, 2018]  Thrun, M. C.: Projection Based Clustering through Self-Organization and Swarm Intelligence, doctoral dissertation 2017, Springer, Heidelberg, ISBN: 978-3-658-20539-3, \doi{10.1007/978-3-658-20540-9}, 2018. 

[Ultsch/Thrun, 2017]  Ultsch, A., & Thrun, M. C.: Credible Visualizations for Planar Projections, in Cottrell, M. (Ed.), 12th International Workshop on Self-Organizing Maps and Learning Vector Quantization, Clustering and Data Visualization (WSOM), IEEE Xplore, France, 2017.
}
\examples{
data("Chainlink")
SampleIdx = sample(1:1000, 200)
Data=Chainlink$Data[SampleIdx,]
Cls=Chainlink$Cls[SampleIdx]
InputDistances=as.matrix(dist(Data))
res=cmdscale(d=InputDistances, k = 2, eig = TRUE, add = FALSE, x.ret = FALSE)
ProjectedPoints=as.matrix(res$points)

# see also R package 'ProjectionBasedClustering' for other common projection methods
# see DatabionicSwarm for projection method without parameters or objective function
# ProjectedPoints=DatabionicSwarm::Pswarm(Data)$ProjectedPoints
\donttest{
resUmatrix=GeneralizedUmatrixGPU(Data,ProjectedPoints)
#library(GeneralizedUmatrix)
#plotTopographicMap(resUmatrix$Umatrix,resUmatrix$Bestmatches,Cls)
}
}
\keyword{GeneralizedUmatrix}
\keyword{U-matrix}
\keyword{Umatrix}
\keyword{U-Matrix}
\keyword{Visualization}
\keyword{DimensionalityReduction}
\keyword{DR}
\keyword{Projection}
\keyword{GPU}
