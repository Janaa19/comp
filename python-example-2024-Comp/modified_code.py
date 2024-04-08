def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model['model']
    classes = dx_model['classes']

    # Extract features.
    features = load_image(record)
    features = np.asarray(features)
    test_data = np.transpose(features, (0, 3, 1, 2)).astype(np.float32)
    images_tensor = torch.tensor(test_data, dtype=torch.float32)
    # Get model probabilities.
    with torch.no_grad():
        # If your model and data are on different devices (e.g., model on GPU), move the data to the same device
        if torch.cuda.is_available():
            images_tensor = images_tensor.to('cuda')
            model.to('cuda')
        
        # Perform prediction
        predictions = model(images_tensor)
        
        # Convert predictions to probabilities using softmax if your model does not include a softmax layer
        probabilities = torch.softmax(predictions, dim=1)
        
        # If you need to move the predictions back to CPU and convert to numpy
        probabilities_np = probabilities.cpu().numpy()
        max_probability = np.argmax(probabilities_np,axis=1)
        labels = [list(classes)[i] for i in max_probability]
    return labels