from gemma_loader import Gemma3nEdge


def build_gloss_sequence(transcript: str, model_name: str, max_tokens: int = 100) -> list[str]:
    """
    Returns a list of uppercase ASL gloss tokens using the selected Gemma model.
    """
    # Initialize Gemma with the chosen model directory
    gemma = Gemma3nEdge(model_dir=model_name)
    # Generate the raw gloss string
    raw_gloss = gemma.generate_gloss(transcript, max_tokens=max_tokens)
    # Split tokens and uppercase for ASL convention
    return [token.upper() for token in raw_gloss.split()]
