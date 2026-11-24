# Run from the repository root: Rscript tests/reference/generate.R
# Base R only. No Python output is used to calculate expected statistics.
options(digits=17)
dest <- "tests/reference"
write_ref <- function(x, name) write.csv(x, file.path(dest, name), row.names=TRUE)
d <- read.csv("examples/students.csv")
d$group <- factor(d$group)
f <- read.csv(file.path(dest, "factorial.csv"))
f$A <- factor(f$A); f$B <- factor(f$B)

for (kind in c("simple", "multiple")) {
  m <- if (kind == "simple") lm(score ~ study_hours, d) else lm(score ~ study_hours + group, d)
  s <- summary(m)
  write_ref(coef(s), paste0("lm_", kind, "_coefficients.csv"))
  write_ref(data.frame(F=s$fstatistic[1], df_model=s$fstatistic[2], df_residual=s$fstatistic[3],
                      p=pf(s$fstatistic[1],s$fstatistic[2],s$fstatistic[3],lower.tail=FALSE),
                      r_squared=s$r.squared, adjusted_r_squared=s$adj.r.squared), paste0("lm_",kind,"_model.csv"))
}
m <- glm(passed ~ study_hours + group, d, family=binomial())
write_ref(coef(summary(m)), "logistic_coefficients.csv")
lr <- m$null.deviance-m$deviance
write_ref(data.frame(LR_chi2=lr, df=m$df.null-m$df.residual,
                    p=pchisq(lr,m$df.null-m$df.residual,lower.tail=FALSE), AIC=AIC(m)), "logistic_model.csv")
write_ref(anova(lm(score ~ group, d)), "oneway.csv")
write_ref(drop1(lm(score ~ group + study_hours, d), test="F"), "ancova.csv")
write_ref(anova(lm(y ~ A * B, f)), "type1.csv")
# Type II: each main effect adjusts for the other main effect, excluding their
# interaction; the interaction adjusts for both main effects.
additive <- drop1(lm(y ~ A+B, f), test="F")
full <- drop1(lm(y ~ A*B, f), test="F")
# Type II uses the full model's residual mean square for all F tests.
ss <- c(additive["A","Sum of Sq"], additive["B","Sum of Sq"], full["A:B","Sum of Sq"])
df <- c(additive["A","Df"], additive["B","Df"], full["A:B","Df"])
mm <- lm(y ~ A*B, f)
ff <- (ss/df)/(deviance(mm)/df.residual(mm))
write_ref(data.frame(effect=c("A","B","A:B"), sum_sq=ss, df=df, F=ff,
                    p=pf(ff,df,df.residual(mm),lower.tail=FALSE)), "type2.csv")
m3 <- lm(y ~ A*B, f, contrasts=list(A=contr.sum, B=contr.sum))
write_ref(drop1(m3, scope=~A+B+A:B, test="F"), "type3.csv")
w <- oneway.test(score ~ group, d, var.equal=FALSE)
write_ref(data.frame(F=unname(w$statistic), df_numerator=w$parameter[1], df_denominator=w$parameter[2],p=w$p.value),"welch.csv")
write_ref(TukeyHSD(aov(score ~ group, d))$group,"tukey.csv")

# Independent Games–Howell reference using R's studentized-range distribution.
lev <- levels(d$group); rows <- list()
for (ij in combn(seq_along(lev),2,simplify=FALSE)) {
  a <- d$score[d$group == lev[ij[1]]]; b <- d$score[d$group == lev[ij[2]]]
  va <- var(a)/length(a); vb <- var(b)/length(b)
  df <- (va+vb)^2/(va^2/(length(a)-1)+vb^2/(length(b)-1))
  est <- mean(a)-mean(b)
  half <- qtukey(.95,length(lev),df)*sqrt((va+vb)/2)
  rows[[length(rows)+1]] <- data.frame(group_a=lev[ij[1]],group_b=lev[ij[2]],estimate=est,
       p_adjusted=ptukey(abs(est)/sqrt((va+vb)/2),length(lev),df,lower.tail=FALSE),ci_low=est-half,ci_high=est+half)
}
write_ref(do.call(rbind,rows),"games_howell.csv")

# OLS pairwise contrasts, with explicit equal-weight reference grids and
# covariate means, verify the marginal-mean implementation independently.
anc <- lm(score ~ group + study_hours, d)
grid <- data.frame(group=factor(lev,levels=lev),study_hours=mean(d$study_hours))
X <- model.matrix(~group+study_hours,grid)
pairs <- combn(seq_along(lev),2,simplify=FALSE)
rows <- lapply(pairs,function(ij) {
  cc <- X[ij[1],]-X[ij[2],]
  est <- as.numeric(cc%*%coef(anc)); se <- sqrt(as.numeric(cc%*%vcov(anc)%*%cc))
  data.frame(group_a=lev[ij[1]],group_b=lev[ij[2]],estimate=est,std_error=se,
             p_raw=2*pt(-abs(est/se),df.residual(anc)))
})
base <- do.call(rbind,rows)
for (method in c("holm","bonferroni","BH","BY")) {
  out <- base; out$p_adjusted <- p.adjust(out$p_raw,method=method)
  write_ref(out,paste0("ancova_posthoc_",method,".csv"))
}

ct <- table(d$group,d$passed)
chi <- chisq.test(ct,correct=FALSE)
write_ref(data.frame(chi2=unname(chi$statistic),df=unname(chi$parameter),p=chi$p.value),"chisq.csv")
gof <- chisq.test(c(20,30,50),p=c(.25,.25,.5))
write_ref(data.frame(chi2=unname(gof$statistic),df=unname(gof$parameter),p=gof$p.value),"gof.csv")

extract_aov <- function(fit) {
  pieces <- summary(fit); out <- list()
  for (stratum in pieces) {
    tab <- stratum[[1]]
    rn <- trimws(rownames(tab))
    residual <- which(rn == "Residuals")
    for (i in which(rn != "Residuals")) {
      out[[length(out)+1]] <- data.frame(effect=rn[i],F=tab[i,"F value"],p=tab[i,"Pr(>F)"],
                                        df_numerator=tab[i,"Df"],df_denominator=tab[residual,"Df"])
    }
  }
  do.call(rbind,out)
}
long <- read.csv("examples/longitudinal.csv")
long$student_id <- factor(long$student_id);long$time <- factor(long$time);long$group <- factor(long$group)
write_ref(extract_aov(aov(score ~ time*group + Error(student_id/time),long)),"mixed.csv")
write_ref(extract_aov(aov(score ~ time + Error(student_id/time),long)),"repeated_one.csv")
two <- read.csv(file.path(dest,"repeated_two.csv"))
two$student_id <- factor(two$student_id);two$time <- factor(two$time);two$measure <- factor(two$measure)
write_ref(extract_aov(aov(score ~ time*measure + Error(student_id/(time*measure)),two)),"repeated_two.csv.expected")

# Greenhouse–Geisser epsilon from orthonormal within-subject contrasts.
epsilon <- function(Z) {
  S <- cov(Z); q <- ncol(Z)
  min(1,sum(diag(S))^2/(q*sum(S*S)))
}
H <- function(k) qr.Q(qr(contr.helmert(k)))
wide <- as.matrix(xtabs(score ~ student_id + time, long))
within_residual <- residuals(lm(wide ~ group,data=long[!duplicated(long$student_id),]))
mixed_eps <- epsilon(within_residual%*%H(3))
one_eps <- epsilon(wide%*%H(3))
arr <- xtabs(score ~ student_id + time + measure,two)
mat <- matrix(arr,nrow=dim(arr)[1])
eps <- c(epsilon(mat%*%kronecker(matrix(rep(1/sqrt(2),2)),H(3))),
         epsilon(mat%*%kronecker(H(2),matrix(rep(1/sqrt(3),3)))),
         epsilon(mat%*%kronecker(H(2),H(3))))
write_ref(data.frame(design=c("mixed","repeated_one","two_time","two_measure","two_interaction"),
                    epsilon=c(mixed_eps,one_eps,eps)),"gg_epsilon.csv")
writeLines(sub("[[:space:]]+$", "", c(R.version.string, capture.output(sessionInfo()))),file.path(dest,"R-session.txt"))
