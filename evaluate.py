def model_eval():
    for idx, batch in enumerate(valid_loader):
        image = batch['image'].to(device=device, dtype=torch.float)
        mask  = batch['mask'].to(device=device, dtype=torch.float)

        with torch.no_grad():
            preds = model(image)

        # post processing VOXEL REMOVE?

        # resize
        image_depth = batch['original_shape'][2].item()
        image_height= batch['original_shape'][0].item()
        image_width = batch['original_shape'][1].item()    

        image_=image_resize_3D(image.squeeze().cpu().detach().numpy(),image_depth,image_height,image_width)

        preds[preds<=0.5] = 0
        preds[preds>0.5] = 1

        if preds.shape[1] > 1:
            print('multi?')
            preds_=image_resize_3D(label_onehot_decode(preds.squeeze().cpu().detach().numpy()),image_depth,image_height,image_width)
            mask_ =image_resize_3D(label_onehot_decode(mask.squeeze().cpu().detach().numpy()),image_depth,image_height,image_width)
        else:
            print('single?')
            preds_=image_resize_3D(preds.squeeze().cpu().detach().numpy(),image_depth,image_height,image_width)
            mask_ =image_resize_3D(mask.squeeze().cpu().detach().numpy(),image_depth,image_height,image_width)

    #     metric_scores_summary(preds.squeeze(),mask.squeeze(),0.3)
    #     metric_scores_summary(preds,mask,0.5)

        image_save_nii(image_,'Output/'+str(batch['filename'][0])+'_image.nii.gz')
        image_save_nii(preds_,'Output/'+str(batch['filename'][0])+'_preds.nii.gz')
        image_save_nii(mask_,'Output/'+str(batch['filename'][0])+'_mask.nii.gz')
        torch.cuda.empty_cache()
