Paper: https://arxiv.org/pdf/2008.11354v2.pdf
Link: https://paperswithcode.com/dataset/brush
Dataset link: https://github.com/brownvc/decoupled-style-descriptors


Concept: By seeing someone writes, we can extrapolates / reconstruct / write a text in his style. 
- Online-to-online, genAI

Works by segmenting the writing character-by-character and extracting via LSTM a representation of the writer style for this character, then using this vector (The last LSTM Cell) as a guide to synthethise other letters.

The character-writer descriptor vectors are called DSD (Decoupled style descriptors).

Take advantage of more data for a writer, able to extrapolate new unseen characters from a writer's previous samples.
