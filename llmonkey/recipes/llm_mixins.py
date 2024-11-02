class ConvenienceLLMMixin:
    """A mixin class that provides convenience methods for BaseLLMModel"""

    def generate_short_text(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image=None,
        temperature=0.7,
        max_tokens=None,
        api_key: str = "",
        exact_required_length: int | None = None,
        strip_output: bool = True,
        retries: int = 3,
    ) -> str | None:
        """
        Convenience method to generate a short piece of text in response to a prompt.
        It handles retries and stripping of unwanted characters, especially also
        to make it work with very small models.

        Args:
        user_prompt: The user's prompt. Defaults to an empty string.
        system_prompt: The system's prompt. Defaults to an empty string.
        temperature: The temperature of the model. Defaults to 0.7.
        max_tokens: The maximum number of tokens to generate. Defaults to 150.
        api_key: The API key to use for the provider. Defaults to an empty string.
        exact_required_length: The exact length of the generated text, e.g. when generating
          language codes. Defaults to None.
        strip_output: Whether to strip unwanted characters from the output. Defaults to True.
        retries: The number of retries to attempt. Defaults to 3.

        Returns:
        The generated text, or None if no text was generated.
        """
        text: str | None = None
        while not text and retries > 0:
            retries -= 1
            response = self.generate_prompt_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                image=image,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            text = response.conversation[-1].content
            if strip_output:
                text = text.strip().strip("```").strip('"').strip("'").strip()
            if exact_required_length and len(text) != exact_required_length:
                text = None
        return text
