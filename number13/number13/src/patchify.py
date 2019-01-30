import numpy as np
import cv2

try:
    xrange
except NameError:
    xrange = range


class PatchGenerator:
    def __init__(self,stepsize, imsize, winsize=None):
        self.stepsize = stepsize
        self.imsize = imsize
        if len(self.imsize)==2:
            self.imsize=(imsize[0],imsize[1],1)
        if winsize is None:
            self.winsize = self.stepsize
        else:
            self.winsize=winsize
        assert self.stepsize==self.winsize # have not debuged different stepsize  and winsize so this assert here 
        self.coords = self.generate_coords()

    def generate_coords(self):
        coords = {}
        count=0
        for x in xrange(0,self.imsize[0], self.winsize):
            for y in xrange(0, self.imsize[1], self.winsize):
                if (y + self.winsize) > self.imsize[1]:
                    y -= (y+self.winsize)-self.imsize[1]
                if (x + self.winsize) > self.imsize[0]:
                    x -= (x+ self.winsize) - self.imsize[0]
                x,y=np.clip(x,0,self.imsize[0]-self.stepsize),np.clip(y,0,self.imsize[1]- self.stepsize)
                coords[count]=[x,x + self.stepsize, y, y + self.stepsize]
                count += 1
        return coords

    def create(self, img,mask=None,nonzero=False,standarize=False,coords=None):
        self.img_patches=[]
        self.mask_patches=[]
        self.img_org=img
        if coords is None:
            coords=self.coords
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        if mask is not None:
            assert mask.shape[0:2]==img.shape[0:2]
        if mask is not None and len(mask.shape)==2:
            mask=mask[:,:,np.newaxis]
        keep_coords={}
        for i, key in enumerate(coords.keys()):
            x1, x2, y1, y2 = coords[key]
            has_nonzero = True
            if mask is not None:
                m=mask[x1:x2,y1:y2,:]
                if nonzero:
                    if m.shape[-1]>1:
                        has_nonzero = (m[:,:,:]>0).any() 
                    else:
                        has_nonzero = (m > 0).any()
                if has_nonzero:
                    keep_coords[key]=coords[key]
                    self.mask_patches.append(m)
            if has_nonzero:
                if standarize:
                    im=img[x1:x2, y1:y2, :].astype('float32')
                    im0=im[im>0]
                    mean,std=im0.mean(),im0.std()
                    im=(im-mean)/std
                    self.img_patches.append(im)
                else:
                    self.img_patches.append(img[x1:x2, y1:y2, :])


        self.img_patches=np.asarray(self.img_patches)
        self.mask_patches=np.asarray(self.mask_patches)
        return self.img_patches,self.mask_patches,keep_coords

    def reconstruct(self,patches,resize=None):
        self.img_re = np.zeros((self.imsize[0],self.imsize[1],patches.shape[-1]),dtype=patches.dtype)
        for i,key in enumerate(self.coords.keys()):
            x1,x2,y1,y2=self.coords[key]
            item=patches[i]
            if resize is not None:
                item=cv2.resize(item,resize)
                if len(item.shape)==2:
                    item=item[:,:,np.newaxsi]
            self.img_re[x1:x2,y1:y2,:]=item
        return self.img_re

    def get_coords(self):
        return self.coords



def tests():
    img = np.zeros((900,900,3))
    p=PatchGenerator(512,img.shape)
    for co in p.get_coords().values():
        print (co,co[1]-co[0],co[3]-co[2])
    patches,_,_ = p.create(img)
    print (patches.shape)

if __name__=='__main__':
    tests()
