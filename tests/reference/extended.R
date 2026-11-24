# Optional extended references: Rscript tests/reference/extended.R
# Uses mvtnorm for independent multivariate-t Dunnett probabilities.
options(digits=17)
dest <- "tests/reference"
d <- read.csv("examples/students.csv"); d$group <- factor(d$group)
lev <- levels(d$group); model <- lm(score~group,d)
df <- df.residual(model); mse <- deviance(model)/df
means <- tapply(d$score,d$group,mean); ns <- table(d$group)
rows <- lapply(combn(lev,2,simplify=FALSE),function(pair) {
  estimate <- means[pair[1]]-means[pair[2]]
  se <- sqrt(mse*(1/ns[pair[1]]+1/ns[pair[2]]))
  critical <- sqrt((length(lev)-1)*qf(.95,length(lev)-1,df))
  data.frame(group_a=pair[1],group_b=pair[2],estimate=unname(estimate),std_error=unname(se),
             p_adjusted=unname(pf((estimate/se)^2/(length(lev)-1),length(lev)-1,df,lower.tail=FALSE)),
             ci_low=unname(estimate-critical*se),ci_high=unname(estimate+critical*se))
})
write.csv(do.call(rbind,rows),file.path(dest,"scheffe.csv"),row.names=FALSE)

library(mvtnorm)
control <- "control"; treatments <- setdiff(lev,control)
se <- sqrt(mse*(1/ns[treatments]+1/ns[control]))
t <- (means[treatments]-means[control])/se
rho <- outer(treatments,treatments,Vectorize(function(a,b) {
  if(a==b) 1 else (1/ns[control])/sqrt((1/ns[a]+1/ns[control])*(1/ns[b]+1/ns[control]))
}))
prob <- function(q) {
  set.seed(101)
  as.numeric(pmvt(lower=rep(-q,length(t)),upper=rep(q,length(t)),df=df,corr=rho,
                  algorithm=GenzBretz(maxpts=1000000,abseps=1e-10,releps=0)))
}
p <- vapply(abs(t),function(v) 1-prob(v),numeric(1))
critical <- uniroot(function(q) prob(q)-.95,c(1,5),tol=1e-8)$root
write.csv(data.frame(group_a=treatments,group_b=control,estimate=as.numeric(means[treatments]-means[control]),
                     p_adjusted=p,ci_low=as.numeric(means[treatments]-means[control]-critical*se),
                     ci_high=as.numeric(means[treatments]-means[control]+critical*se)),
          file.path(dest,"dunnett.csv"),row.names=FALSE)
writeLines(c(R.version.string,paste("mvtnorm",packageVersion("mvtnorm"))),file.path(dest,"R-extended-session.txt"))
