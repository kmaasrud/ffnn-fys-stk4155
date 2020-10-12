

# Syntax {#sec:syntax}

The syntax is a superset of the Markdown syntax, which I'm sure you all know. In addition, Pandoc's Markdown includes some extra features:

- Math formatting:

    This is done inline with single `$`-signs, like this: $\frac{1}{2}$, and display math is done with double dollar signs `$$`.
    
    $$1+1 = 2$$

- Footnotes:

    Are added using a `^` and square brackets, for example ^[Hello, you were sent down here from the text, how cool is that?], which was made by typing `^[Hello, you were sent down here from the text, how cool is that?]`.
    
- Cross-referencing:

    You can cross reference by appending an equation, figure or similar with `{#eq:my-id}` or similar. This is then referenced in the text with `@eq:my-id`. For example, this section is labelled with `{#sec:syntax}`, which i can reference by typing `@sec:syntax`, like this: section @sec:syntax. Cross-referencing requires an extension, the installation of which is described in section @sec:install-xnos
    
- ...and lots more