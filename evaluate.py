
def model_eval():
  for idx, batch in tqdm_notebook(enumerate(valid_loader)):
      image = batch['inputs'].to(device=device, dtype=torch.float)
      mask  = batch['labels'].to(device=device, dtype=torch.float)

      with torch.no_grad():
          preds = model(image)

      # resize
      image_depth = batch['original_shape'][0].item()
      image_height= batch['original_shape'][1].item()
      image_width = batch['original_shape'][2].item()

      image_=image_resize_3D(image.squeeze().cpu().detach().numpy(),image_depth,image_height,image_width)
      preds_=image_resize_3D(preds.squeeze().cpu().detach().numpy(),image_depth,image_height,image_width)
      mask_ =image_resize_3D(mask.squeeze().cpu().detach().numpy(),image_depth,image_height,image_width)

      # post processing VOXEL REMOVE?

      metric_scores_summary(preds_.squeeze(),mask_.squeeze(),threshold=0.3,print_score=True)

      print(idx,batch['filename'],'mask',np.unique(mask_,return_counts=True))
      print(idx,batch['filename'],'pred',np.unique(label_threshold(preds_,0.3),return_counts=True))

  #     image_save_nii(label_threshold(preds_,0.44),'Output/'+str(batch['filename'][0])+'_preds.nii.gz')
  #     image_save_nii(mask_,'Output/'+str(batch['filename'][0])+'_mask.nii.gz')
      torch.cuda.empty_cache()
