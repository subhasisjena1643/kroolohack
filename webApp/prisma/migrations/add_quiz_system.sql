-- Add Quiz and QuizResponse tables

CREATE TABLE IF NOT EXISTS "Quiz" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "question" TEXT NOT NULL,
    "options" TEXT NOT NULL, -- JSON string of options array
    "type" TEXT NOT NULL, -- 'quiz' or 'poll'
    "duration" INTEGER NOT NULL DEFAULT 60, -- duration in seconds
    "status" TEXT NOT NULL DEFAULT 'ACTIVE', -- 'ACTIVE', 'ENDED'
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "endedAt" DATETIME,
    FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "QuizResponse" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "quizId" TEXT NOT NULL,
    "studentId" TEXT NOT NULL, -- anonymous student identifier
    "answer" TEXT NOT NULL,
    "submittedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("quizId") REFERENCES "Quiz" ("id") ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS "Quiz_sessionId_idx" ON "Quiz"("sessionId");
CREATE INDEX IF NOT EXISTS "Quiz_status_idx" ON "Quiz"("status");
CREATE INDEX IF NOT EXISTS "QuizResponse_quizId_idx" ON "QuizResponse"("quizId");
CREATE INDEX IF NOT EXISTS "QuizResponse_studentId_idx" ON "QuizResponse"("studentId");
