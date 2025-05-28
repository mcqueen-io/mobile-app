# Voice Recognition Mobile App

A mobile application with advanced voice recognition capabilities, including noise suppression and speaker identification.

## Features

- Voice recording with noise suppression
- Speaker identification using voice embeddings
- Voice feature extraction using Resemblyzer
- Speaker registration and verification
- Test suite for voice recognition functionality

## Project Structure

```
mobile-app/
├── app/
│   ├── modules/
│   │   └── voice_layer/
│   │       ├── voice_processor.py
│   │       └── test_voice_recognition.py
│   └── config/
│       └── config.template.py
├── models/
│   └── vosk-model-small-en-us/
├── pretrained_models/
├── test_recordings/
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mobile-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configuration:
   - Copy `app/config/config.template.py` to `app/config/config.py`
   - Update the configuration values in `config.py` with your settings
   - Never commit `config.py` to version control

## Usage

Run the voice recognition test:
```bash
python -m app.modules.voice_layer.test_voice_recognition
```

## Development

- The project uses Python 3.8+
- Voice processing is handled by the `voice_processor.py` module
- Test recordings are stored in the `test_recordings` directory
- Pretrained models are stored in the `pretrained_models` directory

## Security

- API keys and sensitive data should be stored in `config.py`
- Never commit `config.py` to version control
- Use environment variables for sensitive data in production

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 