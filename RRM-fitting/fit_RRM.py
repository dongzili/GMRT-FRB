## optimize U, with only Q and V, cable delay without phase

import numpy as np
import numpy.ma as ma
from numpy import cos,sin
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from numpy import newaxis as nax
from astropy import constants as const
import astropy.units as u



#################################################################
#------------------------------------------------
class FitGeneralizedFaraday:
    '''
    fit RRM
    '''
    def __init__(self,freqarr,numSubBand=1):
        '''
        INPUT:
            an array of data frequency
        OPTIONAL:
            number of sub band, default:1

        assign parameter: freqArr, scaledLambdaSquare, scaledFreqArr
        the latter two array calculated respect to the center of the band,
        used for fitting
        '''
        numChannels=len(freqarr)
        self.numSubBand=numSubBand
        self.freqArr=freqarr
        self.wavelength=(const.c/self.freqArr).to(u.m)
        self.scaledLambdaCube=(const.c/self.freqArr)**3\
                -(const.c/self.freqArr[numChannels//2])**3

    def _test_data_dimension(self,data):
        ''' last dim of the IQUV data should be freq'''
        if data.shape[-1]!=len(self.freqArr):
            print('input data dim %d does not match the numChannels specified %d')%(data.shape[-1],len(self.freqArr))
            return -1
        else:
            return 0

    #generate rotation matrix
    def rot_y(self,theta):
        return np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
    def rot_z(self,theta):
        return np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])

    def rot2newEigen(self,stokes,alpha,theta=90*u.deg):
        '''Assume the new eigen axis has polar angle theta and rotation angle alpha. 
        (With quasi linear states, theta=90deg, with quasi-circular, theta=0)
        rotate the stokes parameter to the new coordinate where axis z is in the direction of the new eigen axis.
        INPUT: stokes, with QUV in the last axis'''
        #return np.dot(rot_y(-theta),np.dot(rot_z(-alpha),stokes))
        return np.linalg.multi_dot([self.rot_y(-theta),self.rot_z(-alpha),stokes])

    def rotBackNewEigen(self,stokes,alpha,theta=90*u.deg):
        '''Assume the new eigen axis has polar angle theta and rotation angle alpha. 
        rotate the stokes parameter from the new coordinate where axis z is in the direction of the new eigen axis, 
        back to the normal axis where Q,U,V are x,y,z axis
        INPUT: stokes, with QUV in the last axis'''
        return np.linalg.multi_dot([self.rot_z(alpha),self.rot_y(theta),stokes])


    def rot_around_eigen(self,stokes,phi,alpha,theta=90*u.deg):
        '''Assume the new eigen axis has polar angle theta and rotation angle alpha. 
        rotate the input stokes vector around the eigen axis for phi.
        INPUT: stokes, with QUV in the last axis, angles with unit'''
        stokesP=self.rot2newEigen(stokes,alpha,theta=theta)
        stokesProt=np.dot(self.rot_z(phi),stokesP)
        stokesRot=self.rotBackNewEigen(stokesProt,alpha,theta=theta)
        return stokesRot
    

    def rot_back_QUV(self,pars,QUV,absolute=0):
        RRM=pars[0]*u.rad/u.m**3;alpha=pars[1]*u.deg
        if absolute==0:
            phiArr=RRM*self.scaledLambdaCube
        else:
            phiArr=RRM*self.wavelength**3

        QUVrot=np.zeros(QUV.shape,dtype=float)
        for i in np.arange(QUV.shape[-1]):
            QUVrot[...,i]=self.rot_around_eigen(QUV[...,i],-phiArr[i],alpha,theta=90*u.deg)
        return QUVrot    
    
    def blackman_smooth(self,I,weightsI=None,smWidth=3.):
            freqReso=np.abs(np.diff(self.freqArr[:2])).to(u.MHz).value
            half=np.array([0.297,0.703])
            wdBin=smWidth/(half[1]-half[0])/freqReso

            window=np.blackman(wdBin)
            window/=window.sum()
            if weightsI is None:
                Ismt=np.convolve(I,window,mode='same')
            else: 
                Ismt=np.convolve(I*weightsI,window,mode='same')
                renorm=np.convolve(weightsI,window,mode='same')
                renorm[renorm==0]=1e5
                Ismt/=renorm
                
            return Ismt

    def _loss_function(self,pars,QUV,weight,IsmtRnm):
        '''the function to minimize during the fitting'''
        rottedQUV=self.rot_back_QUV(pars,QUV)

        avQUV=rottedQUV.mean(-1,keepdims=True)*IsmtRnm[nax,:]
        pol=[0,1,2]
        if weight is None:
            distance= (np.abs(rottedQUV[pol]-avQUV[pol])).ravel() ##flatten QUV
        else:
            distance= (np.abs(rottedQUV[pol]-avQUV[pol])*weight[pol]).ravel() ##flatten QUV
        return distance

    def fit_rrm(self,pInit,IQUV,maxfev=20000,ftol=1e-3,IQUVerr=None,bounds=(-np.inf,np.inf),method='trf',smWidth=3.,weights=None):
        '''fitting RRM:
        INPUT:
            initial parameter: pInit=np.array([RRM,stokesPositionAngle]) #stokesPolarAngle set to 90deg
                               RRM: relativistic rotation measure
            IQUV: 4 by len(freq) array

        OPTIONAL:
            weight: an array with the same length as the input frequency, default weight=1.
            parameters for least_squares:
            maxfev,ftol: parameters for leastsq function, default: maxfev=20000,ftol=1e-3
            bounds:default:(-np.inf,np.inf)
            method: default:'trf'
        '''
        if self._test_data_dimension(IQUV)!=0:
            return -1
        if IQUVerr is None:
            weightsI=None
            weightsQUV=None
        else:
            weight=1./IQUVerr
            weight=ma.masked_invalid(weight)
            weight.set_fill_value(0)
            weights=ma.copy(weight)/weight.std()
            weightsI,weightsQUV=weights[0]**2,weights[1:]

        I,QUV=ma.copy(IQUV[0]),ma.copy(IQUV[1:])
        Ismt=self.blackman_smooth(I,weightsI=weightsI,smWidth=smWidth)
        IsmtRnm=Ismt/Ismt.mean()
        
        if weights is not None:
            weightsQUV=np.repeat(weights[None,:],3,axis=0)

        paramFit = least_squares(self._loss_function,pInit,args=(QUV,weightsQUV,IsmtRnm),max_nfev=maxfev,ftol=ftol,bounds=bounds,method=method)      
        para,jac=paramFit.x,paramFit.jac
        rottedQUV=self.rot_back_QUV(para,QUV)
        #return para,jac         
        cov = np.linalg.inv(jac.T.dot(jac))
        paraErr = np.sqrt(np.diagonal(cov))
        print('fitting results para, err',para,paraErr)
        return para,paraErr,rottedQUV

    def show_fitting(self,fitPars,QUV,\
                     I=None,returnPlot=0,fmt='.',title='',xlim=None,pol=[0,1,2],fBin=1):
        '''show QUV matrix before fitting and the QUV after corrected with the fitted parameters
           INPUT:
            pars:the output parameter from fit_rm_cable_delay, it has the same format as pInit
            QUV: the 3 by len(freq) array you used to feed into fit_rm_cable_delay
        '''
        pars=fitPars
        rottedQUV=self.rot_back_QUV(pars,QUV)
        labels=['Q','U','V']
        fig,axes=plt.subplots(2,1,figsize=[14,8],sharex=True,sharey=True)

        if I is not None:
            for i in np.arange(2):
                axes[i].plot(self.freqArr.reshape(-1,fBin).mean(-1),I.reshape(-1,fBin).mean(-1),fmt,color='k',label='I')

        for i in np.arange(2):
            for j in pol:
                if i==0:
                    axes[i].plot(self.freqArr.reshape(-1,fBin).mean(-1),QUV[j].reshape(-1,fBin).mean(-1),fmt,label=labels[j])
                else:
                    axes[i].plot(self.freqArr.reshape(-1,fBin).mean(-1),rottedQUV[j].reshape(-1,fBin).mean(-1),fmt,label='rotted '+labels[j])
            axes[i].legend()
            axes[i].axhline(y=0,color='k')
        axes[0].set_title(title)
        if xlim is not None:
            axes[0].set_xlim(xlim)
        if returnPlot==1:
                return fig,axes
        plt.show()

    def show_derotated(self,fitPars,QUV,\
                     I=None,returnPlot=0,fmt='.',title='',xlim=None,pol=[0,1,2],fBin=1):
        '''show QUV matrix before fitting and the QUV after corrected with the fitted parameters
           INPUT:
            pars:the output parameter from fit_rm_cable_delay, it has the same format as pInit
            QUV: the 3 by len(freq) array you used to feed into fit_rm_cable_delay
        '''
        pars=fitPars
        rottedQUV=self.rot_back_QUV(pars,np.copy(QUV))
        labels=['Q','U','V']
        fig,axes=plt.subplots(1,1,figsize=[14,4],sharex=True,sharey=True)

        if I is not None:
            axes.plot(self.freqArr.reshape(-1,fBin).mean(-1),I.reshape(-1,fBin).mean(-1),fmt,color='k',label='I')

        for j in pol:
                    axes.plot(self.freqArr.reshape(-1,fBin).mean(-1),rottedQUV[j].reshape(-1,fBin).mean(-1),fmt,label='rotted '+labels[j])
        axes.legend()
        axes.axhline(y=0,color='k')
        axes.set_title(title)
        if xlim is not None:
            axes.set_xlim(xlim)
        if returnPlot==1:
                return fig,axes
        plt.show()


