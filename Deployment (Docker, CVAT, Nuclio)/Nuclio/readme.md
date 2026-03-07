# Serverless Model Deployment using Nuclio for CVAT Annotation

Nuclio serverless function for **Defect segmentation in CVAT**.  
The function loads a trained **U-Net model** and returns **CVAT-compatible mask annotations** for uploaded images.

## Files

- `function.yaml` – Nuclio function configuration and build setup  
- `main.py` – Handler for inference 
- `model_handler.py` – Model loading and prediction logic  

## Model

- Architecture: **U-Net**
- Framework: **AtomAI**
- Output: **Segmentation mask**
- Label: `defect`

The trained model (`model_new.pth`) is downloaded during build using **gdown**.

## Deployment

```bash
nuctl deploy --project-name cvat --path . --file function.yaml
```

After deployment, the function will appear in **CVAT → Annotation**.

## Notes

- Ensure the **Google Drive model link is publicly accessible**.
- The model file must download as `    .pth` file.
