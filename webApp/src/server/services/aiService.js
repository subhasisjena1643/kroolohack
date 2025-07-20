const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize Google AI
const genAI = new GoogleGenerativeAI('AIzaSyCSkpRICQgpLopUiB__Vl4BsNuyOptKxZY');

class AIService {
  constructor() {
    this.model = genAI.getGenerativeModel({ model: 'gemini-pro' });
  }

  async summarizeAudioTranscript(audioText, sessionInfo = {}) {
    try {
      const prompt = `
        You are an AI assistant helping students catch up on missed classroom sessions.
        
        Please summarize the following classroom audio transcript into key learning points:
        
        Session: ${sessionInfo.name || 'Classroom Session'}
        Date: ${sessionInfo.date || new Date().toLocaleDateString()}
        Duration: ${sessionInfo.duration || 'Unknown'}
        
        Audio Transcript:
        "${audioText}"
        
        Please provide:
        1. Main topics covered (bullet points)
        2. Key concepts explained
        3. Important announcements or assignments
        4. Questions discussed
        
        Keep the summary concise but comprehensive for a student who missed the class.
      `;

      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error summarizing audio:', error);
      return 'Unable to generate summary at this time.';
    }
  }

  async generateDemoSummary(sessionName, date) {
    try {
      const prompt = `
        Generate a detailed, comprehensive classroom session summary for a student who missed the class.

        Session: ${sessionName}
        Date: ${date}

        Create a DETAILED summary (at least 300-400 words) that includes:

        1. **Detailed Topic Coverage**: Explain each main topic covered in depth, including specific concepts, theories, and examples discussed
        2. **Key Learning Objectives**: What students were expected to learn and understand
        3. **Important Definitions**: Define key terms and concepts introduced
        4. **Examples and Case Studies**: Specific examples, experiments, or case studies discussed
        5. **Class Activities**: Any hands-on activities, group work, or demonstrations performed
        6. **Assignments and Homework**: Detailed description of any assignments given, due dates, and requirements
        7. **Upcoming Content**: Preview of what will be covered in the next session
        8. **Important Announcements**: Any administrative or course-related announcements

        Make it comprehensive and detailed so a student who missed the class can fully understand what they missed.
        Use proper paragraphs and detailed explanations, not just bullet points.
      `;

      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error generating demo summary:', error);
      return `
        ## Session Summary - ${sessionName}

        **Overview**
        Today's session provided a comprehensive introduction to cellular biology, focusing on the fundamental structures and processes that govern life at the microscopic level. The class began with an exploration of cell theory and progressed through detailed examinations of organelles, cellular respiration, and metabolic pathways.

        **Main Topics Covered**

        **Cell Structure and Organization**
        We began by reviewing the basic principles of cell theory, established by Schleiden, Schwann, and Virchow. Students learned that all living organisms are composed of one or more cells, cells are the basic unit of life, and all cells arise from pre-existing cells. We then examined the differences between prokaryotic and eukaryotic cells, with particular emphasis on the presence of membrane-bound organelles in eukaryotes.

        **Mitochondria - The Powerhouse of the Cell**
        A significant portion of the session was dedicated to understanding mitochondrial structure and function. We discussed the double membrane system, cristae formation, and the role of mitochondria in ATP production through cellular respiration. Students learned about the electron transport chain and how the mitochondrial matrix serves as the site for the citric acid cycle.

        **Cellular Respiration Process**
        The class covered the three main stages of cellular respiration: glycolysis, the citric acid cycle, and oxidative phosphorylation. We worked through the chemical equations and energy yields at each stage, emphasizing how glucose is systematically broken down to produce ATP, the cell's energy currency.

        **Laboratory Component**
        Students participated in a hands-on microscopy lab where they observed prepared slides of various cell types, identifying key organelles and structures discussed in the lecture. This practical component reinforced theoretical concepts and improved microscopy skills.

        **Assignment Details**
        For homework, students are required to complete Chapter 7 exercises (questions 1-15) and prepare a detailed diagram of a typical animal cell, labeling all major organelles and their functions. This assignment is due next Tuesday and will count toward the midterm grade.

        **Next Session Preview**
        Our next class will focus on photosynthesis and chloroplast structure, building on today's discussion of cellular energy processes. We'll explore how plants convert light energy into chemical energy and compare this process to cellular respiration.
      `;
    }
  }
}

module.exports = new AIService();
