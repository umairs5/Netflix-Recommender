# Netflix Recommendation System 🎬

A multimodal recommendation system that suggests Netflix shows and movies based on text descriptions and/or images using semantic similarity search powered by transformer models.

## Features

- **Multimodal Input**: Accept text queries, images, or both simultaneously
- **Semantic Search**: Uses sentence transformers for intelligent content matching
- **Image Captioning**: Automatically generates descriptions from uploaded images
- **Flexible Output**: Choose between 3, 5, 10, or 25 recommendations
- **Interactive Interface**: Clean Gradio web interface for easy interaction

## How It Works

1. **Data Processing**: Netflix dataset is processed using SentenceTransformers to create embeddings
2. **Multimodal Input**: 
   - Text queries are directly encoded
   - Images are processed using BLIP (Bootstrapped Language-Image Pre-training) to generate captions
   - Combined inputs merge image captions with text descriptions
3. **Similarity Search**: Cosine similarity is used to find the most relevant Netflix content
4. **Results**: Returns top-N recommendations with title, description, and genre information

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/umairs5/Netflix-Recommender.git
cd Netflix-Recommender
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download and process the Netflix dataset:
```bash
python main.ipynb
```

4. Launch the application:
```bash
python app.py
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
gradio
numpy
pandas
scikit-learn
sentence-transformers
torch
torchvision
transformers
Pillow
kagglehub
```

## Usage

### Running the Application

1. Start the Gradio interface:
```bash
python app.py
```

2. Open your browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`)

3. Select your input type:
   - **Image**: Upload an image to get recommendations based on visual content
   - **Text**: Enter a text description of what you're looking for
   - **Both**: Combine image and text inputs for more precise recommendations

4. Choose the number of recommendations (3, 5, 10, or 25)

5. Click "Submit" to get your personalized Netflix recommendations!

### Example Queries

**Text Input Examples:**
- "Romantic comedy with strong female lead"
- "Dark psychological thriller series"
- "Family-friendly animated movies"
- "British crime drama"

**Image Input:**
- Upload movie posters, scenes, or any relevant images
- The system will generate captions and find similar content

## Project Structure

```
netflix-recommender/
│
├── app.py                 # Main Gradio application
├── main.ipynb            # Data processing and embedding generation
├── requirements.txt      # Python dependencies
├── netflix_embeddings.npy   # Generated embeddings (created after running main.ipynb)
├── netflix_metadata.csv     # Processed metadata (created after running main.ipynb)
└── README.md
```

## Technical Details

### Models Used

- **SentenceTransformer**: `all-MiniLM-L6-v2` for text embeddings
- **BLIP**: `Salesforce/blip-image-captioning-base` for image captioning
- **Similarity Metric**: Cosine similarity for content matching

### Data Processing

The system processes the Netflix dataset by:
1. Combining title, description, and genre information
2. Generating embeddings for the combined text using SentenceTransformers
3. Storing embeddings and metadata for efficient similarity search

### Performance

- **Dataset Size**: ~8,000+ Netflix titles
- **Embedding Dimension**: 384 (MiniLM-L6-v2)
- **Search Time**: <1 second for typical queries
- **Memory Usage**: ~50MB for embeddings

## Customization

### Adding New Data Sources

To use a different dataset:

1. Modify the data loading section in `main.ipynb`
2. Update the column names in the `combine_description_title_and_genre` function
3. Regenerate embeddings by running the notebook

### Changing Models

To use different transformer models:

```python
# In main.ipynb and app.py
model = SentenceTransformer("your-preferred-model")
```

Popular alternatives:
- `all-mpnet-base-v2` (higher quality, slower)
- `all-distilroberta-v1` (balanced performance)
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingual support)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Add user rating integration
- [ ] Implement collaborative filtering
- [ ] Support for video input
- [ ] Multi-language support
- [ ] API endpoint for external integration
- [ ] Recommendation explanation features
- [ ] User preference learning

## Acknowledgments

- Netflix dataset from [Kaggle](https://www.kaggle.com/datasets/infamouscoder/dataset-netflix-shows)
- [SentenceTransformers](https://www.sbert.net/) for semantic embeddings
- [BLIP](https://github.com/salesforce/BLIP) for image captioning
- [Gradio](https://gradio.app/) for the web interface

---

⭐ Star this repository if you found it helpful!
