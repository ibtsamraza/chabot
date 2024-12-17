from pdf2image import convert_from_path
import os
def pdf_to_images(pdf_path, output_folder):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    # Save images to the specified output folder
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.jpg"
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    
    return image_paths

# Example usage
pdf_path = '1 - Data Visualization.pdf'
output_folder = 'output_images'



output_folder = '/home/ibtsam/django_cahtbotapp/aichatbot/intelliDocs/api/output_images/'

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
image_paths = pdf_to_images(pdf_path, output_folder)


