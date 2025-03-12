# Gemma 3 - Multimodal AI Model

This is a [Replicate](https://replicate.com) [Cog](https://github.com/replicate/cog) model for Google's Gemma 3, a powerful multimodal AI model that can process both text and images to generate high-quality text responses.

## Model Description

Gemma 3 is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. This implementation uses the 4B instruction-tuned (IT) version, which is optimized for following instructions and generating helpful responses.

Key features:
- **Multimodal capabilities**: Process both text and images
- **128K context window**: Handle long inputs
- **Multilingual support**: Works with over 140 languages
- **Instruction-tuned**: Optimized for helpful, accurate responses

## Example

When given an image of a bee on a flower with the prompt "Describe this image in detail", Gemma 3 generates a detailed description:

```
Here's a detailed description of the image:

Overall Impression:
The image is a close-up, vibrant photograph of a garden scene featuring pink cosmos flowers and a bumblebee. It has a natural, slightly soft focus, giving it a gentle and appealing aesthetic.

Foreground:
- Cosmos Flowers: The main focus is a cluster of pink cosmos flowers. They have delicate, slightly ruffled petals with a light pink hue. The centers of the flowers are a bright yellow with a dark brown/black marking.
- Bumblebee: A bumblebee is prominently positioned on one of the cosmos flowers, seemingly collecting nectar. It has the characteristic black and yellow stripes of a bumblebee.
- Dried Flowers/Seed Heads: There are several dried, brown seed heads and remnants of spent flowers interspersed among the vibrant cosmos. These add texture and a sense of the natural cycle of growth and decay.
- Greenery: A few green leaves and stems are visible, providing a backdrop for the flowers.

Background:
- Green Leaf: A large, broad green leaf is visible in the background, slightly out of focus. It provides a contrasting color and adds depth to the image.
- Red Flowers: A few small red flowers are visible in the lower right corner.

Color Palette:
The dominant colors are pink, yellow, green, and brown. The pink of the cosmos is the most striking, contrasted by the yellow of the flower centers and the bumblebee.

Lighting:
The lighting appears to be soft and natural, suggesting an overcast day or a shaded area in the garden.

Overall, the image is a beautiful depiction of a moment in a garden, highlighting the interaction between a bumblebee and a flower.
```

## Installation

To run this model locally, you'll need to have [Cog](https://github.com/replicate/cog) installed:

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

Then, clone this repository and build the model:

```bash
git clone https://github.com/yourusername/gemma-3-cog.git
cd gemma-3-cog
cog build
```

## Usage

### Local Usage

Once you've built the model, you can run predictions locally:

```bash
# Text-only query
cog predict -i prompt="What is the capital of France?"

# Image + text query
cog predict -i prompt="Describe this image in detail" -i image=@bee.jpg

```

### API Usage

If you've deployed this model to Replicate, you can use it via the API:

```python
import replicate

output = replicate.run(
    "username/gemma-3:latest",
    input={
        "prompt": "Describe this image in detail",
        "image": open("bee.jpg", "rb"),
        "temperature": 0.7,
        "max_new_tokens": 512
    }
)
print(output)
```

## Model Details

- **Model Family**: Gemma 3
- **Size**: 4B parameters
- **Context Window**: 128K tokens
- **Training Data**: Trained on a diverse dataset including web documents, code, mathematics, and images
- **License**: [Gemma License](https://ai.google.dev/gemma/terms)

## Acknowledgments

This implementation is based on Google's Gemma 3 model. For more information, see the [Gemma 3 Technical Report](https://goo.gle/Gemma3Report).

## License

This model implementation is subject to the [Gemma License](https://ai.google.dev/gemma/terms). Please review the license terms before using this model. 