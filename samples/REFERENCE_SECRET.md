# Sample Image Reference (FOR YOUR EYES ONLY - Model cannot see this!)
# Use this to check if model responses match the actual image content

| Code | Actual Content |
|------|---------------|
| sample_001.jpg | Two cats sleeping on a pink/red couch with TV remotes |
| sample_002.jpg | Banana and chocolate donut |
| sample_003.jpg | Living room with TV, dining table, woman standing, yellow walls |
| sample_004.jpg | Kitchen with wooden cabinets, oranges on table |
| sample_005.jpg | Skate park with graffiti, person on bike, person skateboarding |
| sample_006.jpg | People sitting in dark restaurant/bar |
| sample_007.png | Golden retriever dog sitting in park on grass |
| sample_008.png | Red apple on wooden table |
| sample_009.png | White cat sleeping on blue sofa |

## How to verify model is learning:

1. During training, model will be asked "What do you see in this image?" for these samples
2. Check if the response matches the actual content above
3. If model says "I see a dog" for sample_007.png → Good! Model is learning
4. If model says random gibberish → Model is NOT learning properly

## Expected good responses:
- sample_001.jpg → Should mention "cats", "couch", "sleeping", "remotes"
- sample_007.png → Should mention "dog", "golden retriever", "grass", "park"
- sample_008.png → Should mention "apple", "red", "table", "wooden"
- sample_009.png → Should mention "cat", "white", "sleeping", "sofa/couch"

## Red flags (model not learning):
- Repetitive responses ("I see an image", "This is a photo")
- Completely wrong objects ("I see a car" when it's a cat)
- Gibberish or random tokens
