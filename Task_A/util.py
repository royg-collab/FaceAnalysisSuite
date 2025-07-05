#Plot Curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='train'), plt.plot(history['val_loss'], label='val')
plt.title('Loss'), plt.legend()
plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='train'), plt.plot(history['val_acc'], label='val')
plt.title('Accuracy'), plt.legend()
plt.show()

#Grad-CAM Visualization
import numpy as np
import cv2
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.grad, self.activations = model, None, None
        layer.register_forward_hook(lambda m,inp,out: setattr(self,'activations',out))
        layer.register_backward_hook(lambda m,gi,go: setattr(self,'grad',go[0]))

    def __call__(self, x, cls):
        self.model.zero_grad()
        logits = self.model(x.unsqueeze(0).to(device))
        logits[0,cls].backward()
        grad = self.grad[0].cpu().data
        act  = self.activations[0].cpu().data
        weights = grad.mean(dim=[1,2], keepdim=True)
        cam = np.maximum((weights * act).sum(dim=0).numpy(),0)
        cam = cv2.resize(cam,(224,224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# Example on one image
img, label = val_ds[25]
cam = GradCAM(model, model.se)(img, cls=label)
plt.imshow(np.transpose(img.numpy(),(1,2,0)))
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title(class_names[label]), plt.axis('off')
plt.show()

