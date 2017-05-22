# ============ Import Data ===============
getwd()
setwd("/Users/cindy/Downloads")
ticdata2000<-read.table("ticdata2000.txt",header=FALSE,sep="\t") #training
ticeval2000<-read.table("ticeval2000.txt",header=FALSE,sep='\t') #test
tictgts2000<-read.table("tictgts2000.txt",header=FALSE,sep="\t") #test label

# =========== Exploratory Data Analysis ==============

counts <- table(ticdata2000[,4])
barplot(counts,col="green",names.arg=c("1: 20-30 years", "2: 30-40 years", "3: 40-50 years", "4: 50-60 years", "5: 60-70 years", "6: 70-80 years"))
title(main="Barplot for Levels of Average Age",col.main="blue",font.main=4)

counts <- table(ticdata2000[,47])
barplot(counts,col="green",names.arg=c("Level=0","Level=4","Level=5","Level=6","Level=7","Level=8"))
title(main="Barplot for Levels of Car Policy Contribution",col.main="blue",font.main=4)

counts <- table(ticdata2000[,68])
barplot(counts,col="green",names.arg=c("0 Car Policy","1 Car Policy","2 Car Policy","3 Car Policy","4 Car Policy","6 Car Policy","7 Car Policy"))
title(main="Barplot for Number of Car Policy",col.main="blue",font.main=4)

par(mfrow=c(1,2))
counts1 <- table(ticdata2000[,86])
barplot(counts1,col="red",names.arg=c("Y=0:No Caravan Insurance Policy","Y=1:Having Caravan Insurance Policy"),xlab="Number of Caravan Insurance Policy",ylab="Count",col.lab=rgb(0,0.5,0),cex.names=0.7)
title(main="Distribution of Target Var(Y) on Training Set",col.main="blue",font.main=4)
legend("topright", c("Y=1%: 6%"), cex=0.9)

counts2 <- table(tictgts2000)
barplot(counts2,col="red",names.arg=c("Y=0:No Caravan Insurance Policy","Y=1:Having Caravan Insurance Policy"),xlab="Number of Caravan Insurance Policy",ylab="Count",col.lab=rgb(0,0.5,0),cex.names=0.7)
title(main="Distribution of Target Var(Y) on Test Set",col.main="blue",font.main=4)
legend("topright", c("Y=1%: 6%"), cex=0.9)

 
# ============  
library(infotheo)

# vector MI contains the mutual informations for each feature
MI<-rep(0,85)
for(i in 1:85){
	MI[i]<-mutinformation(ticdata2000[,i],ticdata2000[,86],method="emp")
}

MI.percent<-MI/MI[which.max(MI)]

# vector o contains the ordering according to MI
o<-order(MI,decreasing=TRUE)
var.name<-names(ticdata2000)[-86]

# data.frame MI.sort.df contains:feature names,mutual information and percentage,sorted from high to low
MI.sort.df<-data.frame(var=var.name[o],mutualinfo=MI[o],percent=MI.percent[o])

#plot feature importance
mycolor<-c(rep("red",10),rep("hotpink",10),rep("pink",10),rep("yellow",10),rep("lightskyblue",10),rep("slateblue",10),rep("gray",25))
barplot(MI.sort.df$percent,col=mycolor,xlab="Feature",ylab="Percentage of IG relative to Best Feature ",col.lab=rgb(0,0.5,0))
title(main="Predictive Power by Information Gain",col.main="blue",font.main=4)

#top 10 features
MI.sort.df$var[1:10]






 # ===================Make Training and Test Sets==============================================
tic.train.x<-ticdata2000[,o]
tic.train.y<-ticdata2000[,86]
tic.test.x<-ticeval2000[,o]
tic.test.y <-tictgts2000$V1

fold1 <- c(1:2911)
fold2 <- c(2912:5822)






# ========================= Naive Bayes Classification Function=======================================================
NaiveBayes <- function(Xtrain,Ytrain,Xtest,d){
	
	#class probability y=0
    prob.y0<-length(which(Ytrain==0))/length(Ytrain)

    #class probability y=1  
    prob.y1<-length(which(Ytrain==1))/length(Ytrain) 
    
    # predicted value vector for test set
    Ytest<-rep(0,dim(Xtest)[1])

    # log probability matrices for two class
    log.prob.y0 <- mat.or.vec(dim(Xtest)[1],d)
    log.prob.y1 <- mat.or.vec(dim(Xtest)[1],d)

	for(i in 1:d){
		# extract possible values for each feature from both training and test sets
        values<-unique(c(Xtrain[,i],Xtest[,i]))
        
        # mulinomial parameters for feature i,row 1 for class y=0,row 2 for class y=1
       par.est<-mat.or.vec(2,length(values))  
       
       for(j in 1:length(values)){
	   par.est[1,j]<-(1+length(which(Ytrain==0 & Xtrain[,i]==values[j])))/(length(values)+length(which(Ytrain==0)))
	   par.est[2,j]<-(1+length(which(Ytrain==1 & Xtrain[,i]==values[j])))/(length(values)+length(which(Ytrain==1)))
       }
       
       for(obs in 1:dim(Xtest)[1]){
       log.prob.y0[obs,i]<-log(par.est[1,which(values==Xtest[obs,i])])
       log.prob.y1[obs,i]<-log(par.est[2,which(values==Xtest[obs,i])])
	   }
	 }
	   
	 prob.class0<-rowSums(log.prob.y0)+log(prob.y0)
     prob.class1<-rowSums(log.prob.y1)+log(prob.y1)

     for(h in 1:dim(Xtest)[1]){if(prob.class1[h]>=prob.class0[h]){Ytest[h]<-1}}
    
	return(Ytest)
}

tr.accuracy <-rep(0,17)
te.accuracy <-rep(0,17)
sens <-rep(0,17)
spec <-rep(0,17)
prec<-rep(0,17)
F1 <-rep(0,17)

npar <-seq(from=5,to=85,by=5)

for(n in npar){

	y1 <- NaiveBayes(tic.train.x[fold1,],tic.train.y[fold1],tic.train.x,n)
	
	y2 <- NaiveBayes(tic.train.x[fold2,],tic.train.y[fold2],tic.train.x,n)

	# train accuracy
	tr.a1 <- (length(which(y1[fold1]==1&tic.train.y[fold1]==1))+length(which(y1[fold1]==0&tic.train.y[fold1]==0)))/length(fold1)
	tr.a2 <- (length(which(y2[fold2]==1&tic.train.y[fold2]==1))+length(which(y2[fold2]==0&tic.train.y[fold2]==0)))/length(fold2)
	tr.accuracy[n/5] <-(tr.a1+tr.a2)/2
	
	# test accuracy
	te.a1 <- (length(which(y1[fold2]==1&tic.train.y[fold2]==1))+length(which(y1[fold2]==0&tic.train.y[fold2]==0)))/length(fold2)
	te.a2 <- (length(which(y2[fold1]==1&tic.train.y[fold1]==1))+length(which(y2[fold1]==0&tic.train.y[fold1]==0)))/length(fold1)
	te.accuracy[n/5] <- (te.a1+te.a2)/2
	
	# sensitivity
	sens1 <- length(which(y1[fold2]==1&tic.train.y[fold2]==1))/length(which(tic.train.y[fold2]==1))
	sens2 <- length(which(y2[fold1]==1&tic.train.y[fold1]==1))/length(which(tic.train.y[fold1]==1))
	sens[n/5]<-(sens1+sens2)/2
	
	# specificity
	spec1 <- length(which(y1[fold2]==0&tic.train.y[fold2]==0))/length(which(tic.train.y[fold2]==0))
	spec2 <- length(which(y2[fold1]==0&tic.train.y[fold1]==0))/length(which(tic.train.y[fold1]==0))
	spec[n/5] <-(spec1+spec2)/2
	
	# precision
	prec1 <- length(which(y1[fold2]==1&tic.train.y[fold2]==1))/length(which(y1[fold2]==1))
	prec2 <- length(which(y2[fold1]==1&tic.train.y[fold1]==1))/length(which(y2[fold1]==1))
	prec[n/5]<-(prec1+prec2)/2
	
	# F1-score
	F1.1<- 2*(prec1*sens1)/(prec1+sens1)
	F1.2<- 2*(prec2*sens2)/(prec2+sens2)
	F1[n/5]<-(F1.1+F1.2)/2
    
}

tr.accuracy <-c(0.9266575, 0.8928203, 0.8668842, 0.8473033, 0.8376846, 0.8265201, 0.8251460, 0.8290965, 0.8261766, 0.8268636, 0.8277224, 0.8290965, 0.8280660, 0.8289248, 0.8306424, 0.8311577, 0.8320165)
te.accuracy <-c(0.9223634,0.8907592,0.8588114,0.8461010,0.8364823,0.8227413,0.8187908,0.8187908,0.8151838,0.8160426,0.8151838,0.8163861,0.8187908,0.8187908,0.8201649,0.8223978,0.8239437)
sens <- c(0.1577461, 0.3076123, 0.3650015, 0.4108931, 0.4196650, 0.4452869, 0.4308653, 0.4336902, 0.4221925, 0.4222916, 0.4194667, 0.4195659, 0.4193676, 0.4165428, 0.4165428, 0.4165428, 0.4165428)
spec <- c(0.9709558, 0.9278540, 0.8902275, 0.8737854, 0.8630085, 0.8467460, 0.8434566, 0.8432741, 0.8401695, 0.8410855, 0.8403552, 0.8416325, 0.8441905, 0.8443722, 0.8458348, 0.8482091, 0.8498522)
prec <- c(0.2577108, 0.2168567, 0.1777221, 0.1738759, 0.1650017, 0.1572394, 0.1500701, 0.1508329, 0.1450922, 0.1460919, 0.1446679, 0.1453682, 0.1477266, 0.1468128, 0.1482043, 0.1501185, 0.1513811)
F1 <-c(0.1956434, 0.2529388, 0.2378792, 0.2435536, 0.2360746, 0.2320185, 0.2222945, 0.2235105, 0.2156370, 0.2166310, 0.2146663, 0.2154863, 0.2180505, 0.2167018, 0.2181574, 0.2202426, 0.2216375)

# accuracy plot
plot(te.accuracy,type="b",col="red",xlab="Number of Splits",axes=FALSE,ylim=range(c(te.accuracy,tr.accuracy)),ylab="Train/Test Accuracy %",col.lab=rgb(0,0.5,0))
lines(tr.accuracy,type="b",col="green")
title(main="Train and Test Accuracy",col.main="blue",font.main=4)
axis(1,at=1:length(npar),labels=npar)
axis(2,at=c(0.8,0.825,0.85,0.875,0.9,0.925,0.95),labels=c("80%","82.5%","85%","87.5%","90%","92.5%","95%"))
legend("topright", c("Test Accuracy","Training Accuracy"), cex=0.6, 
   col=c("red","green"), lty=1)

# sensitivity,specificity,precision plot
plot(sens,type="b",col="red",xlab="Number of Splits",axes=FALSE,ylim=range(c(sens,spec,prec)),ylab="Sensitivity/Specificity/Precision %",col.lab=rgb(0,0.5,0))
lines(spec,type="b",col="green")
lines(prec,type="b",col="purple")
title(main="Sensitivity/Specificity/Precision",col.main="blue",font.main=4)
axis(1,at=1:length(npar),labels=npar)
axis(2,at=seq(from=0.1,to=1,by=0.1),labels=c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"))
legend("topright", c("Sensitivity","Specificity","Precision"), cex=0.6, 
   col=c("red","green","purple"), lty=1)

# F1 plot
plot(F1,type="b",col="red",xlab="Number of Splits",axes=FALSE,ylim=range(c(F1)),ylab="F1-Score",col.lab=rgb(0,0.5,0))
title(main="F1-Score",col.main="blue",font.main=4)
axis(1,at=1:length(npar),labels=npar)
axis(2,at=seq(from=0.15,to=0.275,by=0.025),labels=seq(from=0.15,to=0.275,by=0.025))

# Test Set
y <- NaiveBayes(tic.train.x,tic.train.y,tic.test.x,10)
table(y,tic.test.y)
sensitivity<-length(which(y==1&tic.test.y==1))/length(which(tic.test.y==1))
specificity<-length(which(y==0&tic.test.y==0))/length(which(tic.test.y==0))
precision<-length(which(y==1&tic.test.y==1))/length(which(y==1))
F1<-2*(precision*sensitivity)/(precision+sensitivity)


 



# ======================SVM=================================
library(class)
library(e1071)

# =================2-Fold Cross Validation:RBF Kernels======================================

# Tuning Parameter: cost, gamma
 
tr.accuracy <-mat.or.vec(4,7)
te.accuracy <-mat.or.vec(4,7)
sens <-mat.or.vec(4,7)
spec <-mat.or.vec(4,7)
prec<-mat.or.vec(4,7)
F1 <-mat.or.vec(4,7)

for(g in c(0.01,0.1,1,10)){
for(c in c(1,10,100,1000,10000,100000,1000000)){

	model1<-svm(tic.train.x[fold1,1:10],tic.train.y[fold1],type="C",kernel="radial",cost=c,gamma=g)
	y1<-predict(model1,tic.train.x[,1:10])
	 
	model2<-svm(tic.train.x[fold2,1:10],tic.train.y[fold2],type="C",kernel="radial",cost=c,gamma=g)
	y2<-predict(model2,tic.train.x[,1:10])
	
	# train accuracy
	tr.a1 <- (length(which(y1[fold1]==1&tic.train.y[fold1]==1))+length(which(y1[fold1]==0&tic.train.y[fold1]==0)))/length(fold1)
	tr.a2 <- (length(which(y2[fold2]==1&tic.train.y[fold2]==1))+length(which(y2[fold2]==0&tic.train.y[fold2]==0)))/length(fold2)
	tr.accuracy[log10(g)+3,log10(c)+1] <-(tr.a1+tr.a2)/2
	
	# test accuracy
	te.a1 <- (length(which(y1[fold2]==1&tic.train.y[fold2]==1))+length(which(y1[fold2]==0&tic.train.y[fold2]==0)))/length(fold2)
	te.a2 <- (length(which(y2[fold1]==1&tic.train.y[fold1]==1))+length(which(y2[fold1]==0&tic.train.y[fold1]==0)))/length(fold1)
	te.accuracy[log10(g)+3,log10(c)+1] <- (te.a1+te.a2)/2
	
	# sensitivity
	sens1 <- length(which(y1[fold2]==1&tic.train.y[fold2]==1))/length(which(tic.train.y[fold2]==1))
	sens2 <- length(which(y2[fold1]==1&tic.train.y[fold1]==1))/length(which(tic.train.y[fold1]==1))
	sens[log10(g)+3,log10(c)+1]<-(sens1+sens2)/2
	
	# specificity
	spec1 <- length(which(y1[fold2]==0&tic.train.y[fold2]==0))/length(which(tic.train.y[fold2]==0))
	spec2 <- length(which(y2[fold1]==0&tic.train.y[fold1]==0))/length(which(tic.train.y[fold1]==0))
	spec[log10(g)+3,log10(c)+1] <-(spec1+spec2)/2
	
	# precision
	prec1 <- length(which(y1[fold2]==1&tic.train.y[fold2]==1))/length(which(y1[fold2]==1))
	prec2 <- length(which(y2[fold1]==1&tic.train.y[fold1]==1))/length(which(y2[fold1]==1))
	prec[log10(g)+3,log10(c)+1]<-(prec1+prec2)/2
	
	# F1-score
	F1.1<- 2*(prec1*sens1)/(prec1+sens1)
	F1.2<- 2*(prec2*sens2)/(prec2+sens2)
	F1[log10(g)+3,log10(c)+1]<-(F1.1+F1.2)/2
	 
}
}


 
tr.accuracy<-matrix(c(0.9402267, 0.9402267, 0.9402267, 0.9402267, 0.9424596, 0.9505325, 0.9668499, 0.9402267, 0.9421161, 0.9565441, 0.9750945, 0.9854002, 0.9866025, 0.9866025, 0.9457231, 0.9833391, 0.9866025, 0.9866025, 0.9866025, 0.9866025, 0.9866025, 0.9843696, 0.9866025, 0.9866025, 0.9866025, 0.9866025, 0.9866025, 0.9866025),nrow=4,ncol=7,byrow=TRUE)
te.accuracy<-matrix(c(0.9402267, 0.9402267, 0.9402267, 0.9402267, 0.9378221, 0.9268293, 0.9039849, 0.9402267, 0.9386809, 0.9230505, 0.8988320, 0.8807970, 0.8765029, 0.8753006, 0.9390244, 0.9111989, 0.9041566, 0.9041566, 0.9041566, 0.9041566, 0.9041566, 0.9328409, 0.9318104, 0.9318104, 0.9318104, 0.9318104, 0.9318104, 0.9318104),nrow=4,ncol=7,byrow=TRUE)
sens<- matrix(c(0.000000000, 0.000000000, 0.00000000, 0.00000000, 0.002923977, 0.03731787, 0.11448112, 0.000000000, 0.002824859, 0.04871642, 0.14030132, 0.160769155, 0.16389137, 0.16389137, 0.002824859, 0.095004460, 0.10055506, 0.10055506, 0.100555060, 0.10055506, 0.10055506, 0.031767271, 0.034691248, 0.03469125, 0.03469125, 0.034691248, 0.03469125, 0.03469125),nrow=4,ncol=7,byrow=TRUE)
spec<-matrix(c(1.0000000, 1.0000000, 1.0000000, 1.0000000, 0.9972628, 0.9833769, 0.9541414, 1.0000000, 0.9981744, 0.9786292, 0.9470194, 0.9265634, 0.9218173, 0.9205371, 0.9985385, 0.9631074, 0.9552535, 0.9552535, 0.9552535, 0.9552535, 0.9552535, 0.9901372, 0.9888586, 0.9888586, 0.9888586, 0.9888586, 0.9888586, 0.9888586),nrow=4,ncol=7,byrow=TRUE)

prec<-matrix(c(0,0,0,0,0, 0.1255556, 0.1352746, 0, 0.1666667, 0.1299124, 0.1429271, 0.1223564, 0.1177760, 0.1159627, 0.1000000, 0.1440254, 0.1301522, 0.1301522, 0.1301522, 0.1301522, 0.1301522, 0.1666667, 0.1598746, 0.1598746, 0.1598746, 0.1598746, 0.1598746, 0.1598746),nrow=4,ncol=7,byrow=TRUE)

F1<-matrix(c(0,0,0,0,0,0.05750367, 0.12364935, 0, 0, 0.07070533, 0.14149229, 0.13893404, 0.13688121, 0.13571230, 0, 0.11306354, 0.11230077, 0.11230077, 0.11230077, 0.11230077, 0.11230077, 0.05303777, 0.05662678, 0.05662678, 0.05662678, 0.05662678, 0.05662678, 0.05662678),nrow=4,ncol=7,byrow=TRUE)
  
# accuracy plot
plot(tr.accuracy[1,],type="b",col="yellowgreen",xlab="Cost C",axes=FALSE,ylim=range(tr.accuracy),ylab="Training Accuracy %",col.lab=rgb(0,0.5,0))
lines(tr.accuracy[2,],type="b",col="purple")
lines(tr.accuracy[3,],type="b",col="seagreen")
lines(tr.accuracy[4,],type="b",col="hotpink")
title(main="Training Accuracy",col.main="blue",font.main=4)
axis(1,at=1:7,labels=c(1,10,100,1000,10000,100000,1000000))
axis(2,at=c(0.94,0.95,0.96,0.97,0.98,0.99),labels=c("94%","95%","96%","97%","98%","99%"))
legend("bottomright", c("gamma=0.01","gamma=0.1","gamma=1","gamma=10"), cex=0.6, 
   col=c("yellowgreen","purple","seagreen","hotpink"), lty=1)

plot(te.accuracy[1,],type="b",col="yellowgreen",xlab="Cost C",axes=FALSE,ylim=range(te.accuracy),ylab="Test Accuracy %",col.lab=rgb(0,0.5,0))
lines(te.accuracy[2,],type="b",col="purple")
lines(te.accuracy[3,],type="b",col="seagreen")
lines(te.accuracy[4,],type="b",col="hotpink")
title(main="Test Accuracy",col.main="blue",font.main=4)
axis(1,at=1:7,labels=c(1,10,100,1000,10000,100000,1000000))
axis(2,at=c(0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95),labels=c("87%","88%","89%","90%","91%","92%","93%","94%","95%"))
legend("bottomright", c("gamma=0.01","gamma=0.1","gamma=1","gamma=10"), cex=0.6, 
   col=c("yellowgreen","purple","seagreen","hotpink"), lty=1)

# sensitivity,specificity,precision plot
plot(sens[1,],type="b",col="yellowgreen",xlab="Cost C",axes=FALSE,ylim=range(sens),ylab="Sensitivity %",col.lab=rgb(0,0.5,0))
lines(sens[2,],type="b",col="purple")
lines(sens[3,],type="b",col="seagreen")
lines(sens[4,],type="b",col="hotpink")
title(main="Sensitivity",col.main="blue",font.main=4)
axis(1,at=1:7,labels=c(1,10,100,1000,10000,100000,1000000))
axis(2,at=c(0,0.05,0.10,0.15,0.20),labels=c("0%","5%","10%","15%","20%"))
legend("topright", c("gamma=0.01","gamma=0.1","gamma=1","gamma=10"), cex=0.6, 
   col=c("yellowgreen","purple","seagreen","hotpink"), lty=1)

plot(spec[1,],type="b",col="yellowgreen",xlab="Cost C",axes=FALSE,ylim=range(spec),ylab="Specificity %",col.lab=rgb(0,0.5,0))
lines(spec[2,],type="b",col="purple")
lines(spec[3,],type="b",col="seagreen")
lines(spec[4,],type="b",col="hotpink")
title(main="Specificity",col.main="blue",font.main=4)
axis(1,at=1:7,labels=c(1,10,100,1000,10000,100000,1000000))
axis(2,at=c(0.9,0.92,0.94,0.96,0.98,1.00),labels=c("90%","92%","94%","96%","98%","100%"))
legend("topright", c("gamma=0.01","gamma=0.1","gamma=1","gamma=10"), cex=0.6, 
   col=c("yellowgreen","purple","seagreen","hotpink"), lty=1)

plot(prec[1,],type="b",col="yellowgreen",xlab="Cost C",axes=FALSE,ylim=range(prec),ylab="Precision %",col.lab=rgb(0,0.5,0))
lines(prec[2,],type="b",col="purple")
lines(prec[3,],type="b",col="seagreen")
lines(prec[4,],type="b",col="hotpink")
title(main="Precision",col.main="blue",font.main=4)
axis(1,at=1:7,labels=c(1,10,100,1000,10000,100000,1000000))
axis(2,at=c(0,0.04,0.08,0.12,0.16,0.20),labels=c("0%","4%","8%","12%","16%","20%"))
legend("topright", c("gamma=0.01","gamma=0.1","gamma=1","gamma=10"), cex=0.6, 
   col=c("yellowgreen","purple","seagreen","hotpink"), lty=1)



# F1 plot
plot(F1[1,],type="b",col="yellowgreen",xlab="Cost C",axes=FALSE,ylim=range(F1),ylab="F1",col.lab=rgb(0,0.5,0))
lines(F1[2,],type="b",col="purple")
lines(F1[3,],type="b",col="seagreen")
lines(F1[4,],type="b",col="hotpink")
title(main="F1-Score",col.main="blue",font.main=4)
axis(1,at=1:7,labels=c(1,10,100,1000,10000,100000,1000000))
axis(2,at=c(0,0.2,0.4,0.6,0.8,0.10,0.12,0.14,0.16),labels=c(0,0.2,0.4,0.6,0.8,0.10,0.12,0.14,0.16))
legend("topright", c("gamma=0.01","gamma=0.1","gamma=1","gamma=10"), cex=0.6, 
   col=c("yellowgreen","purple","seagreen","hotpink"), lty=1)


# Test Set
model<-svm(tic.train.x[,1:10],tic.train.y,type="C",kernel="radial",cost=1000,gamma=0.1)
y<-predict(model,tic.test.x[,1:10])
table(y,tic.test.y)
sensitivity<-length(which(y==1&tic.test.y==1))/length(which(tic.test.y==1))
specificity<-length(which(y==0&tic.test.y==0))/length(which(tic.test.y==0))
precision<-length(which(y==1&tic.test.y==1))/length(which(y==1))
F1<-2*(precision*sensitivity)/(precision+sensitivity)





# =========================Classification Tree=======================================================
library(rpart)
library(cluster)
library(foreach)
library(lattice)
library(plyr)
library(reshape2)
library(caret)
 

# use all features to build a full tree
fit<-rpart(ticdata2000$V86~.,method="class", y=TRUE,control=rpart.control(cp=0,xval=2), parms=list(split="information"),data=ticdata2000)

#plot tree
plot(fit, margin=0,uniform=T, branch=1)
text(fit, use.n=TRUE, all=TRUE, cex=.5)
title(main="TIC Data using all features",col.main="blue",font.main=4)

# Feature importance
var.imp<-fit$variable.importance
mycolor1<-c(rep("red",10),rep("hotpink",10))
barplot(var.imp[c(1:20)],col=mycolor1,xlab="Feature",ylab="Importance",col.lab=rgb(0,0.5,0))
title(main="Feature Importance by Tree",col.main="blue",font.main=4)

# CP table
printcp(fit)
plotcp(fit,upper="size")

# Prune the tree
num.split<-fit$cptable[,"nsplit"] #possible number of splits
tr.accuracy <-rep(0,length(num.split))
te.accuracy <-rep(0,length(num.split))
sens <-rep(0,length(num.split))
spec <-rep(0,length(num.split))
prec<-rep(0,length(num.split))
F1 <-rep(0,length(num.split))

for(i in 1:length(num.split)){
	fit.prune <- prune(fit,cp=fit$cptable[i,"CP"])
	y.tr<- predict(fit.prune,ticdata2000[,1:85],type="class")
	y.te <- predict(fit.prune,ticeval2000,type="class")
	
	# train accuracy
	tr.accuracy[i] <-length(which(y.tr==tic.train.y))/length(tic.train.y)
	
	# test accuracy
	te.accuracy[i] <- length(which(y.te==tic.test.y))/length(tic.test.y)
	
	# sensitivity
	sens[i]<-length(which(y.te==1&tic.test.y==1))/length(which(tic.test.y==1))
	
	# specificity
	spec[i] <-length(which(y.te==0&tic.test.y==0))/length(which(tic.test.y==0))
	
	# precision
	prec[i]<-length(which(y.te==1&tic.test.y==1))/length(which(y.te==1))
	
	# F1-score
	F1[i]<-2*(prec[i]*sens[i])/(prec[i]+sens[i])
}

prec[1]<-0
F1[1]<-0


# accuracy plot
plot(te.accuracy,type="b",col="red",xlab="Number of Splits",axes=FALSE,ylim=range(c(te.accuracy,tr.accuracy)),ylab="Train/Test Accuracy %",col.lab=rgb(0,0.5,0))
lines(tr.accuracy,type="b",col="green")
title(main="Train and Test Accuracy",col.main="blue",font.main=4)
axis(1,at=1:length(num.split),labels=num.split)
axis(2,at=c(0.90,0.91,0.92,0.93,0.94,0.95),labels=c("90%","91%","92%","93%","94%","95%"))
legend("topleft", c("Test Accuracy","Training Accuracy"), cex=0.8, 
   col=c("red","green"), lty=1)

# sensitivity,specificity,precision plot
plot(sens,type="b",col="red",xlab="Number of Splits",axes=FALSE,ylim=range(c(sens,spec,prec)),ylab="Sensitivity/Specificity/Precision %",col.lab=rgb(0,0.5,0))
lines(spec,type="b",col="green")
lines(prec,type="b",col="purple")
title(main="Sensitivity/Specificity/Precision",col.main="blue",font.main=4)
axis(1,at=1:length(num.split),labels=num.split)
axis(2,at=seq(from=0.1,to=1,by=0.1),labels=c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"))
legend("topleft", c("Sensitivity","Specificity","Precision"), cex=0.6, 
   col=c("red","green","purple"), lty=1)

# F1 plot
plot(F1,type="b",col="red",xlab="Number of Splits",axes=FALSE,ylim=range(c(F1)),ylab="F1-Score",col.lab=rgb(0,0.5,0))
title(main="F1-Score",col.main="blue",font.main=4)
axis(1,at=1:length(num.split),labels=num.split)
axis(2,at=seq(from=0,to=0.16,by=0.04),labels=seq(from=0,to=0.16,by=0.04))

# Test Set
y <- predict(fit,ticeval2000,type="class")
table(y,tic.test.y)
sensitivity<-length(which(y==1&tic.test.y==1))/length(which(tic.test.y==1))
specificity<-length(which(y==0&tic.test.y==0))/length(which(tic.test.y==0))
precision<-length(which(y==1&tic.test.y==1))/length(which(y==1))
F1<-2*(precision*sensitivity)/(precision+sensitivity)


# Pruned tree with nsplit=3
colnames(ticdata2000)[c(47,59,61)] <- c("Contribution car policies","Contribution fire policies","Contribution boat policies")
fit<-rpart(ticdata2000$V86~.,method="class", y=TRUE,control=rpart.control(cp=0,xval=2), parms=list(split="information"),data=ticdata2000)

pruned.tree <- prune(fit,cp=fit$cptable[2,"CP"])

# Plot Pruned Tree
plot(pruned.tree, margin=0.1,uniform=T, branch=1)
text(pruned.tree, use.n=TRUE, all=FALSE, cex=0.8,col="red")
 
 

# =============================== AdaBoost =========================================
library(rpart)
library(ada)

T<-c(1,20,40,60,80,100,500,1000)
tr.accuracy <-rep(0,length(T))
te.accuracy <-rep(0,length(T))
sens <-rep(0,length(T))
spec <-rep(0,length(T))
prec<-rep(0,length(T))
F1 <-rep(0,length(T))

fold1 <- c(1:2911)
fold2 <- c(2912:5822)
data1<-ticdata2000[fold1,]
data2<-ticdata2000[fold2,]


for(t in 1:length(T)){
	model1 <- ada(data1$V86~.,data=data1,iter = T[t], loss = "e", type = "discrete")
	y1 <-predict(model1,ticdata2000[,1:85])	
	model2 <- ada(data2$V86~.,data=data2,iter = T[t], loss = "e", type = "discrete")
	y2 <-predict(model2,ticdata2000[,1:85])	


	# train accuracy
	tr.a1 <- (length(which(y1[fold1]==1&tic.train.y[fold1]==1))+length(which(y1[fold1]==0&tic.train.y[fold1]==0)))/length(fold1)
	tr.a2 <- (length(which(y2[fold2]==1&tic.train.y[fold2]==1))+length(which(y2[fold2]==0&tic.train.y[fold2]==0)))/length(fold2)
	tr.accuracy[t] <-(tr.a1+tr.a2)/2
	
	# test accuracy
	te.a1 <- (length(which(y1[fold2]==1&tic.train.y[fold2]==1))+length(which(y1[fold2]==0&tic.train.y[fold2]==0)))/length(fold2)
	te.a2 <- (length(which(y2[fold1]==1&tic.train.y[fold1]==1))+length(which(y2[fold1]==0&tic.train.y[fold1]==0)))/length(fold1)
	te.accuracy[t] <- (te.a1+te.a2)/2
	
	# sensitivity
	sens1 <- length(which(y1[fold2]==1&tic.train.y[fold2]==1))/length(which(tic.train.y[fold2]==1))
	sens2 <- length(which(y2[fold1]==1&tic.train.y[fold1]==1))/length(which(tic.train.y[fold1]==1))
	sens[t]<-(sens1+sens2)/2
	
	# specificity
	spec1 <- length(which(y1[fold2]==0&tic.train.y[fold2]==0))/length(which(tic.train.y[fold2]==0))
	spec2 <- length(which(y2[fold1]==0&tic.train.y[fold1]==0))/length(which(tic.train.y[fold1]==0))
	spec[t] <-(spec1+spec2)/2
	
	# precision
	prec1 <- length(which(y1[fold2]==1&tic.train.y[fold2]==1))/length(which(y1[fold2]==1))
	prec2 <- length(which(y2[fold1]==1&tic.train.y[fold1]==1))/length(which(y2[fold1]==1))
	prec[t]<-(prec1+prec2)/2
	
	# F1-score
	F1.1<- 2*(prec1*sens1)/(prec1+sens1)
	F1.2<- 2*(prec2*sens2)/(prec2+sens2)
	F1[t]<-(F1.1+F1.2)/2
    

		
}

tr.accuracy <-c(0.9390244, 0.9438337, 0.9445208, 0.9472690, 0.9510477, 0.9534524, 0.9879766, 0.9938166)
te.accuracy <-c(0.9316386, 0.9388526, 0.9393679, 0.9364480, 0.9357609, 0.9343868, 0.9261422, 0.9227070)
sens <-c(0.051838636, 0.008573694, 0.014322529, 0.017345624, 0.011596789, 0.011596789, 0.054960848, 0.054960848)
spec <-c(0.9875840, 0.9979903, 0.9981736, 0.9948861, 0.9945203, 0.9930605, 0.9815521, 0.9779001)
prec <-c(0.23717949, 0.20833333, 0.33928571, 0.17216117, 0.11071429, 0.08928571, 0.15277379, 0.13101604)
F1 <-c(0.08269891, 0.01646053, 0.02747753, 0.03135965, 0.02094241, 0.02031098, 0.08026768, 0.07664332)

# accuracy plot
plot(te.accuracy,type="b",col="red",xlab="Number of Weak Learner",axes=FALSE,ylim=range(c(te.accuracy,tr.accuracy)),ylab="Train/Test Accuracy %",col.lab=rgb(0,0.5,0))
lines(tr.accuracy,type="b",col="green")
title(main="Train and Test Accuracy",col.main="blue",font.main=4)
axis(1,at=1:length(T),labels=T)
axis(2,at=seq(from=0.9,to=1,by=0.02),labels=c("90%","92%","94%","96%","98%","100%"))
legend("topright", c("Test Accuracy","Training Accuracy"), cex=0.8, 
   col=c("red","green"), lty=1)

# sensitivity,specificity,precision plot
plot(sens,type="b",col="red",xlab="Number of Weak Learner",axes=FALSE,ylim=range(c(sens,spec,prec)),ylab="Sensitivity/Specificity/Precision %",col.lab=rgb(0,0.5,0))
lines(spec,type="b",col="green")
lines(prec,type="b",col="purple")
title(main="Sensitivity/Specificity/Precision",col.main="blue",font.main=4)
axis(1,at=1:length(T),labels=T)
axis(2,at=seq(from=0.1,to=1,by=0.1),labels=c("10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"))
legend("topright", c("Sensitivity","Specificity","Precision"), cex=0.8, 
   col=c("red","green","purple"), lty=1)

# F1 plot
plot(F1,type="b",col="red",xlab="Number of Weak Learner",axes=FALSE,ylim=range(c(F1)),ylab="F1-Score",col.lab=rgb(0,0.5,0))
title(main="F1-Score",col.main="blue",font.main=4)
axis(1,at=1:length(T),labels=T)
axis(2,at=seq(from=0,to=0.01,by=0.001),labels=seq(from=0,to=0.01,by=0.001))

# Test Set
model <- ada(ticdata2000$V86~.,data=ticdata2000,iter = 500, loss = "e", type = "discrete")
y <-predict(model,ticeval2000)	

table(y,tic.test.y)
sensitivity<-length(which(y==1&tic.test.y==1))/length(which(tic.test.y==1))
specificity<-length(which(y==0&tic.test.y==0))/length(which(tic.test.y==0))
precision<-length(which(y==1&tic.test.y==1))/length(which(y==1))
F1<-2*(precision*sensitivity)/(precision+sensitivity)




 
