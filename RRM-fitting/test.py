from fit_RRM import *
import numpy as np
import numpy.ma as ma

mockdata=np.load('mock_rrm.npz')
IQUV=mockdata['stokes'];freqArr=mockdata['freqArr']*u.MHz
RRM=mockdata['RRM'];stokesPositionAngle=mockdata['stokesPositionAngle']
labels=['I','Q','U','V']

fit=FitGeneralizedFaraday(freqArr)
rrm,alpha =20,20
pInit=np.array([rrm,alpha])
#pInit=np.array([rm,(tau),tau,tau,tau,phi,phi,phi,phi,psi])

#bounds=(pInit+[-20,-0.1,-5,-5],pInit+[20,0.1,5,5])
para,paraErr,derot=fit.fit_rrm(pInit,IQUV,ftol=1e-5,method='dogbox')#,bounds=bounds)
#rottedQUV=fit.rot_back_QUV(pOut,QUV,numSubBand=numSubBand,power2Q=power2Q)
#fit.show_fitting(pOut.x,QUV,numSubBand=numSubBand,power2Q=1,fmt='--')
fig,axes=fit.show_fitting(para,IQUV[1:],I=IQUV[0],
    fmt='--',\
    title=r'RRM=%.1f $\pm$ %.1f, $\alpha=$%.1f $\pm$ %.1f; input RRM=%.1f, $\alpha=$%.1f'\
                        %(para[0],paraErr[0],para[1],paraErr[1],RRM,stokesPositionAngle),returnPlot=0)
