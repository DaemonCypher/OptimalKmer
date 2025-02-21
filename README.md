# OptimalKmer

K-mer-based feature extraction is a widely used technique for microbial taxonomy classification, 
offering insights into the relationships between DNA sequences and taxonomic labels. The 16S 
rRNA gene database used in this study contains nucleotide sequences of varying lengths (90 to 
564 bases) and was filtered to include only records with complete taxonomic annotations down 
to the species level. The specific challenge addressed here is identifying the optimal k-mer size 
for predicting species-level classification with high accuracy. A neural network model was 
developed to test our hypothesis. Here we show that by systematically evaluating k-mer values 
maximize prediction accuracy across taxonomic levels, with larger k-mer performing better with 
strong correlation between the two. The model highlights the trade-offs between k-mer size, 
computational efficiency, and classification performance, demonstrating that certain k-mer 
values outperform others in predicting taxonomic labels. These findings advance our 
understanding of feature optimization in microbial taxonomy classification and set the stage for 
future genome-based research in this field.
