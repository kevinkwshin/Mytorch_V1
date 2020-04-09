def model_test(model,inputs_shape):
    model.eval()
    try:
        inputs = torch.rand(inputs_shape)
        preds = model(inputs)
        print('inputs_shape',inputs_shape)
        print('shape:',preds.shape)
        print('preds',preds)
    except:
        inputs = torch.rand(inputs_shape).cuda()
        preds = model(inputs).cuda()
        print('inputs_shape',inputs_shape)
        print('shape:',preds.shape)
        print('preds',preds)
