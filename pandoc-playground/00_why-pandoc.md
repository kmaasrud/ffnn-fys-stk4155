---
numbersections: true
---
# Why Pandoc?

[Pandoc](https://pandoc.org/) is a document converter, and it supports a plethora of different formats. In addition, the Pandoc guys have defined their own Markdown flavour, which makes writing documents very easy. I prefer this over LaTeX for its simplicity and speed of creation, but the joy of using Pandoc's Markdown is its support of typing raw LaTeX inline. For example, I can type this directly into the document

```latex
\begin{center}
\begin{tabular}{ c c c }
 cell1 & cell2 & cell3 \\ 
 cell4 & cell5 & cell6 \\  
 cell7 & cell8 & cell9    
\end{tabular}
\end{center}
```

which produces this:

\begin{center}
\begin{tabular}{ c c c }
 cell1 & cell2 & cell3 \\ 
 cell4 & cell5 & cell6 \\  
 cell7 & cell8 & cell9    
\end{tabular}
\end{center}

So generally put, you get the power of LaTeX, but wrapped in a more minimalistic markup, for when the extra features aren't needed.
