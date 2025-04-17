# InternVL_V2 

Vision-language model for receipt and tax document analysis with enhanced question classification, synthetic data generation, and response templates.

## Project Overview

This project extends the InternVL vision-language model for document analysis focusing on three key components:

1. **Enhanced Question Classification**: Accurately classify user questions to select appropriate response templates
2. **Synthetic Data Generation**: Create realistic synthetic receipts and tax documents for training
3. **Template-based Response Generation**: Use structured templates to generate consistent responses

## System Architecture

### Overall System Architecture

```mermaid
graph TD
    User([User]) --> |Question + Image| InputHandler[Input Handler]
    InputHandler --> QuestionClassifier[Question Classifier]
    InputHandler --> InternVL2[InternVL2 Vision Encoder]
    
    QuestionClassifier --> |Question Type| ResponseGen[Response Generator]
    InternVL2 --> |Visual Features| ResponseGen
    
    ResponseGen --> |Question Type: DOCUMENT_TYPE| DocType[Document Type Module]
    ResponseGen --> |Question Type: COUNTING| Counting[Counting Module]
    ResponseGen --> |Question Type: DETAIL_EXTRACTION| Details[Detail Extraction Module]
    ResponseGen --> |Question Type: PAYMENT_INFO| Payment[Payment Info Module]
    ResponseGen --> |Question Type: TAX_INFO| Tax[Tax Info Module]
    
    DocType --> Response[Response Formatter]
    Counting --> Response
    Details --> Response
    Payment --> Response
    Tax --> Response
    
    Response --> User
    
    style InternVL2 fill:#f9d5e5,stroke:#333,stroke-width:2px
    style QuestionClassifier fill:#eeeeee,stroke:#333,stroke-width:2px
    style ResponseGen fill:#d5f9e8,stroke:#333,stroke-width:2px
```

### Training Pipeline Architecture

```mermaid
graph LR
    DataGen[Data Generation] --> |Synthetic Documents| RawData[Raw Data]
    RawData --> |Split| TrainData[Training Data]
    RawData --> |Split| ValData[Validation Data]
    RawData --> |Split| TestData[Test Data]
    
    TrainData --> ClassifierTraining[Question Classifier Training]
    ValData --> ClassifierTraining
    ClassifierTraining --> |Enhanced Classifier| Models[(Trained Models)]
    
    TrainData --> MultimodalTraining[Multimodal Model Training]
    ValData --> MultimodalTraining
    MultimodalTraining --> |InternVL2 Model| Models
    
    Models --> |Load Models| Evaluation[Evaluation]
    TestData --> Evaluation
    
    Orchestrator[Training Orchestrator] --> |Manage| MultimodalTraining
    Monitor[Training Monitor] --> |Visualize| MultimodalTraining
    
    subgraph Training Components
        ClassifierTraining
        MultimodalTraining
        Orchestrator
        Monitor
    end
    
    style DataGen fill:#f9d5e5,stroke:#333,stroke-width:2px
    style MultimodalTraining fill:#d5f9e8,stroke:#333,stroke-width:2px
    style ClassifierTraining fill:#eeeeee,stroke:#333,stroke-width:2px
```

### Question Classification Flow

```mermaid
flowchart TD
    Input[User Question] --> Tokenizer[Tokenizer]
    Tokenizer --> BERT[Language Model Encoder]
    BERT --> |Encoded Text| Classifier[Neural Classifier]
    Classifier --> |Logits| SoftMax[Softmax Layer]
    SoftMax --> |Probability Distribution| Class[Question Type]
    Class --> |Route to Appropriate Handler| Handler[Response Generator]
    
    style BERT fill:#f9d5e5,stroke:#333,stroke-width:2px
    style Classifier fill:#d5f9e8,stroke:#333,stroke-width:2px
```

### Multimodal Processing Architecture

```mermaid
graph TD
    Image[Input Image] --> |Preprocess| VisionEncoder[Vision Encoder]
    Question[Input Question] --> |Tokenize| TextEncoder[Text Encoder]
    
    VisionEncoder --> |Visual Features| Fusion[Cross-Modal Fusion]
    TextEncoder --> |Text Features| Fusion
    
    Fusion --> |Classification Task| Classifier[Document Classifier]
    Fusion --> |Generation Task| Generator[Text Generator]
    Fusion --> |Counting Task| Counter[Receipt Counter]
    
    Classifier --> |Classification Output| Output[Model Output]
    Generator --> |Generation Output| Output
    Counter --> |Counting Output| Output
    
    style VisionEncoder fill:#f9d5e5,stroke:#333,stroke-width:2px
    style TextEncoder fill:#eeeeee,stroke:#333,stroke-width:2px
    style Fusion fill:#d5f9e8,stroke:#333,stroke-width:2px
```

## Directory Structure

The project is organized into the following structure:

- **config/**: Configuration files for all components
  - `classifier/`: Question classifier configurations
  - `data_generation/`: Data generation configurations
  - `model/`: Model training configurations

- **data/**: Data processing and dataset modules
  - `datasets/`: Dataset class definitions
  - `generators/`: Synthetic data generation modules

- **models/**: Model implementations
  - `classification/`: Question classifier models
  - `components/`: Shared model components
  - `vision_language/`: Vision-language model implementations

- **scripts/**: Executable scripts for all tasks
  - `classification/`: Question classification scripts
  - `data_generation/`: Data generation scripts
  - `training/`: Model training and evaluation scripts

- **utils/**: Utility functions used across the project

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/InternVL_V2.git
cd InternVL_V2

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate internvl_env
```

## Usage

### Question Classification

Train the enhanced question classifier:

```bash
PYTHONPATH=. python scripts/classification/train_enhanced_classifier.py --num-epochs 15
```

Test the classifier:

```bash
PYTHONPATH=. python scripts/classification/test_enhanced_classifier.py
```

### Synthetic Data Generation

Generate synthetic receipts and tax documents:

```bash
PYTHONPATH=. python scripts/data_generation/generate_data.py --output_dir datasets/synthetic_receipts
```

### InternVL Model Training

Train the multimodal vision-language model:

```bash
PYTHONPATH=. python scripts/training/train_multimodal.py --config config/model/multimodal_config.yaml
```

Evaluate the model:

```bash
PYTHONPATH=. python scripts/training/evaluate_multimodal.py --model_path models/multimodal/best_model.pt
```

## Components

### Question Classifier

The enhanced question classifier categorizes questions into 5 types:
- DOCUMENT_TYPE: Questions about the type of document
- COUNTING: Questions about counting documents
- DETAIL_EXTRACTION: Questions about specific details in the document
- PAYMENT_INFO: Questions about payment methods
- TAX_INFO: Questions about tax-related information

### Data Generators

The project includes synthetic data generators for:
- Australian receipts with realistic product listings, pricing, and layouts
- Australian tax documents with proper formatting and tax-specific fields

### Vision-Language Model

Based on InternVL, extended with:
- Question classification component
- Template-based response generation
- Detail extraction capabilities

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request