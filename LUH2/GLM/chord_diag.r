library("RColorBrewer")
library("circlize")
library("plyr")
library("shape") # for arrows
library(extrafont)
loadfonts()

rm(list=ls())
base_dir = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Output\\GLM\\LUH2_v1h_beta\\'
file_name = 'df_trans.csv'
col_type = 'Set1'

reg = c('crop',	'pastr',	'primf',	'primn',	'range',	'secdf',	'secdn',	'urban')


num_reg = length(reg)

circ<- matrix(data=0.0,nrow=num_reg,ncol=num_reg,byrow=TRUE)
diag(circ) <- 100.0

matrix1 <-read.table(paste(base_dir,file_name,sep=""), header=T, row.names=1,sep=",")
m <- as.matrix(matrix1)
dimnames(m) <- NULL
tiff(file=paste(base_dir,"Activity_Matrix.tiff",sep=""), family='Arial',width=4,height=4,units="in",res=400) 
# Convert all diagonal elements to 0
#diag(m) <- 0.0
# Put all negative elements to 0
#m[m<0.0] <- 0.0
# Normalize
#reg_sum <- sum(m)
#m <- m*100./reg_sum
# Put all below threshold elements to 0
#m[m<1.0] <- 0.0
rownames(m) = reg
colnames(m) = reg
chordDiagram(m, grid.col=brewer.pal(num_reg,col_type), annotationTrack = "grid",
             preAllocateTracks = list(track.height = 0.3))
circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise",
              niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA) # here set bg.border to NA is important
Arrows(0.56,-0.31,0.34,-0.13,col='black',lwd=1.0,arr.lwd=1.0)

dev.off()