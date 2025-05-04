library(tidyverse)
library(Seurat)
library(data.table)
library(hdWGCNA)

dat <- readRDS("/gpfs/gibbs/pi/gerstein/rm2686/MultiomeSexDifferenceAnalysis/RM-sexDiffAnalysis/allCountsSZBD.rds")
meta <- readRDS("/gpfs/gibbs/pi/gerstein/rm2686/MultiomeSexDifferenceAnalysis/RM-sexDiffAnalysis/allMetaSZBD.rds")

wgcna <- readRDS("/gpfs/gibbs/pi/gerstein/rm2686/MultiomeSexDifferenceAnalysis/RM-sexDiffAnalysis/sczWGCNA/auto/Oligo/p0-mainNetwork.rds")
tom <- GetTOM(wgcna) %>% as.data.frame
metacells <- GetMetacellObject(wgcna)

exprMat <- metacells@assays$RNA@layers$data %>% as.data.frame
rownames(exprMat) <- rownames(metacells)
colnames(exprMat) <- colnames(metacells)

sampleMeta <- meta %>% distinct(sample, .keep_all=T)
exprMeta <- metacells@meta.data
exprMeta$sex <- sampleMeta$sex[match(exprMeta$sample, sampleMeta$sample)]
exprMeta$disorder <- sampleMeta$disorder[match(exprMeta$sample, sampleMeta$sample)]
exprMeta$age <- sampleMeta$age[match(exprMeta$sample, sampleMeta$sample)]
exprMeta$PMI <- sampleMeta$PMI[match(exprMeta$sample, sampleMeta$sample)]

genes <- wgcna@misc$Oligo$wgcna_degrees$gene_name
exprMat <- exprMat[genes,]

fwrite(exprMat, "Documents/cpsc452/final/oligo-SCZ-metacellExpr.csv", row.names = T)
fwrite(exprMeta, "Documents/cpsc452/final/oligo-SCZ-meta.csv", row.names = T)
fwrite(tom, "Documents/cpsc452/final/oligo-SCZ-tom.csv", row.names = T)

test <- fread("Documents/cpsc452/final/oligo-SCZ-metacellExpr.csv", data.table = F)

thresholds <- seq(from=0, to=1, by=0.005)
tomFrac <- c()
for (thr in thresholds) {
  tomFrac <- c(tomFrac, sum(tom > thr) / (nrow(tom)*ncol(tom)))
}
thresholdingVals <- data.frame(threshold=thresholds, frac=tomFrac)
ggplot(thresholdingVals, aes(x=threshold, y=frac)) + geom_line()
