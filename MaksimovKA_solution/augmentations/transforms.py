from albumentations import (
                            Compose,
                            HorizontalFlip,
                            ShiftScaleRotate,
                            OneOf,
                            RandomContrast,
                            RandomGamma,
                            RandomBrightness
   
                            )


def augmentations(prob=0.5):
    
    transformer = Compose([
                      
            HorizontalFlip(p=prob),
            ShiftScaleRotate(p=prob, shift_limit=0.1, scale_limit=.1, rotate_limit=10),
            OneOf([RandomContrast(limit=0.1, p=prob),
                   RandomGamma(gamma_limit=(90, 110), p=prob),
                   RandomBrightness(limit=0.1, p=prob)],p=prob),
            
    ], p=prob)
    return transformer


