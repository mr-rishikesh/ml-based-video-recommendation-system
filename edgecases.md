1.


1. The "Passing Mention" vs. "Deep Explanation" ProblemThe Scenario: A user searches for "How to implement JWT Authentication." * Video A: A 2-hour general backend tutorial where the instructor says, "We will use JWT authentication later, but first let's set up the database." (This is a passing mention).Video B: A 10-minute video entirely about coding a JWT flow.The Edge Case: The text chunk from Video A might actually score higher than Video B because it contains the exact phrase "use JWT authentication", even though the instructor never actually teaches it!The Solution (Density Tracking): Don't just look at a single chunk. Analyze the Topic Density. If a topic is only found in one 2-minute chunk and never mentioned again in the surrounding chunks, penalize its score. If the vectors for Chunk 1, Chunk 2, and Chunk 3 all show high similarity to "JWT", it proves the instructor is actually explaining it deeply.2. The Information Splintering Problem (Cross-Chunk Context)The Scenario: A user searches for "What is the difference between React and Angular?"The Edge Case: In a video comparing the two, the instructor talks entirely about React from minute 2:00 to 4:00. Then, the instructor talks entirely about Angular from 4:00 to 6:00.If your system blindly cuts chunks every 2 minutes, neither chunk contains the comparison. Chunk 1 only knows about React. Chunk 2 only knows about Angular. Your vector database will score both of them poorly because neither chunk answers the user's specific query about the difference between them.The Solution (Semantic Chunking): Stop chunking by arbitrary time (like every 2 minutes). Instead, use an NLP technique called Semantic Boundary Detection. The pipeline reads the transcript and looks for transitional phrases (like "Now let's compare this to..." or "Moving on to the next topic..."). It groups the text by idea, ensuring the context of the comparison stays inside a single vector.3. The Recency vs. Relevance ParadoxThe Scenario: A user searches for "Next.js App Router tutorial."The Edge Case: Video A is a highly popular, perfectly transcribed tutorial from 2022 (using older Next.js 12 technology). Video B is a brand-new tutorial from 2026 (using the current technology), but the instructor's transcript is slightly messier.The AI will recommend Video A because the semantic match is mathematically tighter, completely ignoring the fact that the code inside the video is deprecated and useless to the user today.The Solution (Time-Decay Functions): You must alter the vector database's sorting algorithm. You apply a mathematical Time Decay. If a video is more than 2 years old, its vector similarity score is multiplied by $0.8$. If it is newer than 6 months, it is multiplied by $1.1$. This ensures the database prioritizes semantic meaning, but nudges fresh, accurate technology to the top of the list.











This is the easiest to implement if you are already using a database like Qdrant. You assign mathematical weights to the video's metadata.

You tell the database: "Run the vector search on the transcript chunks. BUT, if the user's query matches the video_title or the video_tags (using standard text search), multiply that chunk's score by 1.5."

Because Video 1 is titled "React Router Tutorial", every single chunk inside it gets an artificial boost, pushing it above the isolated chunk from Video 2.