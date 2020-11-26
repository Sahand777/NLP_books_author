
# =============================================================================
# make a folder to save results
if not os.path.exists('./WordCloud'):
    os.makedirs('WordCloud')
# =============================================================================

# =============================================================================
# Word Cloud Visualization
wordcloud1 = WordCloud().generate(TheThreeMusketeers)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.title("WordCloud of The Three Musketeers")
plt.savefig("./WordCloud/WordCloud of The Three Musketeers.png")
plt.show()

wordcloud2 = WordCloud().generate(SherlockHolmes)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.title("WordCloud of Adventures of Sherlock Holmes")
plt.savefig("./WordCloud/WordCloud of Adventures of Sherlock Holmes.png")
plt.show()

wordcloud3 = WordCloud().generate(Oz)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.title("WordCloud of Dorothy and the Wizard in Oz")
plt.savefig("./WordCloud/WordCloud of Dorothy and the Wizard in Oz.png")
plt.show()

wordcloud4 = WordCloud().generate(MysteriousIsland)
plt.imshow(wordcloud4, interpolation='bilinear')
plt.title("WordCloud of The Mysterious Island")
plt.savefig("./WordCloud/WordCloud of The Mysterious Island.png")
plt.show()

wordcloud5 = WordCloud().generate(MechanicalDrawing)
plt.imshow(wordcloud5, interpolation='bilinear')
plt.title("WordCloud of Mechanical Drawing Self-Taught")
plt.savefig("./WordCloud/WordCloud of Mechanical Drawing Self-Taught.png")
plt.show()

wordcloud6 = WordCloud().generate(ArtofPhotography)
plt.imshow(wordcloud6, interpolation='bilinear')
plt.title("WordCloud of The History and Practice of the Art of Photography")
plt.savefig("./WordCloud/WordCloud of The History and Practice of the Art of Photography.png")
plt.show()

wordcloud7 = WordCloud().generate(ChristmasCarol)
plt.imshow(wordcloud7, interpolation='bilinear')
plt.title("WordCloud of ChristmasCarol, by Charles Dickens")
plt.savefig("./WordCloud/WordCloud of ChristmasCarol, by Charles Dickens.png")
plt.show()


# =============================================================================