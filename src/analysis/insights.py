"""
AI-powered insights generation using Google Gemini.
Analyzes app reviews and market data to generate actionable business insights.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiInsightsGenerator:
    """
    AI-powered insights generator using Google Gemini.
    Analyzes app reviews and market data to provide actionable business insights.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize the Gemini AI client."""
        # Load environment variables
        load_dotenv()

        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Configure Gemini API
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Set generation config for consistent output
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.3,  # Lower temperature for more consistent analysis
            top_p=0.8,
            max_output_tokens=4096,
            response_mime_type="application/json"  # Request JSON output
        )

        logger.info(f"GeminiInsightsGenerator initialized with model: {model_name}")

    def test_connection(self) -> bool:
        """
        Test the Gemini API connection and basic functionality.

        Returns:
            bool: True if connection successful
        """
        logger.info("Testing Gemini API connection...")

        try:
            # Simple test prompt
            test_prompt = "Respond with a JSON object containing {'test': 'success', 'status': 'connected'}"

            response = self.model.generate_content(
                test_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                result = json.loads(response.text)
                if result.get('test') == 'success':
                    logger.info("✅ Gemini API connection test successful")
                    return True
                else:
                    logger.error("❌ Gemini API test failed - unexpected response")
                    return False
            else:
                logger.error("❌ Gemini API test failed - empty response")
                return False

        except Exception as e:
            logger.error(f"❌ Gemini API connection test failed: {e}")
            return False

    def _create_market_analysis_prompt(self, reviews_data: List[Dict], category: str) -> str:
        """
        Create a prompt for market analysis of app reviews.

        Args:
            reviews_data: List of review dictionaries
            category: App category for context

        Returns:
            str: Formatted prompt for the AI
        """
        total_reviews = len(reviews_data)

        numeric_ratings = []
        for review in reviews_data:
            rating_val = review.get('score', review.get('rating', None))
            if rating_val is None:
                continue
            try:
                numeric_ratings.append(float(rating_val))
            except (TypeError, ValueError):
                continue

        average_rating = sum(numeric_ratings) / len(numeric_ratings) if numeric_ratings else 0.0

        # Sample reviews for analysis (limit to avoid token limits)
        sample_reviews = reviews_data[:50] if len(reviews_data) > 50 else reviews_data

        reviews_text = "\n".join([
            f"Review {i+1}: Rating {review.get('score', review.get('rating', 0))}/5 - {review.get('text', review.get('review_text', ''))}"
            for i, review in enumerate(sample_reviews)
        ])

        stats_summary = (
            f"Total reviews collected: {total_reviews}. "
            f"Average rating across all collected reviews: {average_rating:.2f}/5."
        )

        prompt = f"""
You are a senior market analyst for a mobile app company. Analyze the following user reviews for apps in the '{category}' category.

**Overall Review Statistics:**
{stats_summary}

**Reviews to Analyze:**
{reviews_text}

**Analysis Requirements:**
Based on the reviews provided, identify:

1. **TOP 3 MOST REQUESTED FEATURES** (features users are explicitly asking for or missing):
   - For each feature, provide evidence from specific reviews
   - Include confidence score (0.0-1.0) based on how frequently and strongly it's mentioned
   - Provide actionable product manager recommendation

2. **TOP 3 MOST COMMON COMPLAINTS** (pain points and issues users are experiencing):
   - For each complaint, provide evidence from specific reviews
   - Include confidence score (0.0-1.0) based on frequency and severity
   - Provide actionable product manager recommendation

3. **OVERALL SENTIMENT SUMMARY**:
   - Calculate average rating from the provided reviews
   - Identify key themes in positive vs negative feedback
   - Provide market positioning recommendations

**Output Format (Valid JSON only):**
{{
  "category": "{category}",
  "total_reviews_analyzed": {total_reviews},
  "average_rating": {average_rating:.2f},
  "sentiment_summary": "Brief analysis of overall sentiment",
  "requested_features": [
    {{
      "feature": "Feature name",
      "confidence_score": 0.95,
      "evidence": "Quote from review showing this request",
      "recommendation": "Actionable recommendation for product manager"
    }}
  ],
  "common_complaints": [
    {{
      "complaint": "Complaint description",
      "confidence_score": 0.88,
      "evidence": "Quote from review showing this issue",
      "recommendation": "Actionable recommendation for product manager"
    }}
  ]
}}

**Important Guidelines:**
- Base your analysis ONLY on the reviews provided
- Even if only a subset of reviews is shown above, remember that the total_reviews_analyzed field must equal {total_reviews} and average_rating must equal {average_rating:.2f}
- Use specific quotes as evidence for each insight
- Focus on actionable, specific recommendations
- Ensure confidence scores reflect the strength of evidence
- Be professional and data-driven in your analysis
"""

        return prompt

    def generate_insights(self, reviews_data: List[Dict], category: str) -> Optional[Dict[str, Any]]:
        """
        Generate AI-powered insights from app reviews.

        Args:
            reviews_data: List of review dictionaries
            category: App category for context

        Returns:
            Dict containing structured insights or None if failed
        """
        logger.info(f"Generating insights for {category} category with {len(reviews_data)} reviews")

        try:
            # Create the analysis prompt
            prompt = self._create_market_analysis_prompt(reviews_data, category)

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            if not response.text:
                logger.error("Empty response from Gemini API")
                return None

            # Parse JSON response
            try:
                insights = json.loads(response.text)
                logger.info(f"Successfully generated insights for {category}")
                return insights

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response.text[:500]}...")
                return None

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return None

    def _create_d2c_analysis_prompt(self, campaign_data: pd.DataFrame, seo_category: str) -> str:
        """
        Create a prompt for D2C campaign analysis.

        Args:
            campaign_data: DataFrame with D2C campaign data
            seo_category: SEO category for the campaigns

        Returns:
            str: Formatted prompt for the AI
        """
        # Convert campaign data to readable format
        campaign_summary = campaign_data.to_dict('records')

        prompt = f"""
You are a senior digital marketing analyst specializing in D2C e-commerce. Analyze the following ad campaign data for the '{seo_category}' category.

**Campaign Performance Data:**
{json.dumps(campaign_summary, indent=2)}

**Analysis Requirements:**

1. **TOP PERFORMING CAMPAIGNS**:
   - Identify campaigns with highest ROAS (Return on Ad Spend)
   - Identify campaigns with highest conversion rates
   - Analyze what makes these campaigns successful

2. **OPTIMIZATION RECOMMENDATIONS**:
   - Suggest improvements for low-performing campaigns
   - Recommend budget allocation adjustments
   - Provide specific optimization tactics

3. **CREATIVE GENERATION**:
   - Generate 2 high-CTR ad headlines (under 60 characters)
   - Generate 1 SEO-optimized meta description (under 160 characters)
   - Ensure copy is compelling and conversion-focused

4. **STRATEGIC INSIGHTS**:
   - Overall campaign health assessment
   - Market opportunity identification
   - Competitive positioning recommendations

**Output Format (Valid JSON only):**
{{
  "seo_category": "{seo_category}",
  "total_campaigns": {len(campaign_data)},
  "top_performers": [
    {{
      "campaign_name": "Campaign identifier",
      "roas": 0.0,
      "conversion_rate": 0.0,
      "success_factors": "What makes this campaign work"
    }}
  ],
  "optimization_recommendations": [
    {{
      "campaign_type": "Campaign category",
      "current_performance": "Current metrics",
      "recommended_actions": "Specific optimization steps"
    }}
  ],
  "creative_content": {{
    "ad_headlines": [
      "High-converting headline 1",
      "High-converting headline 2"
    ],
    "meta_description": "SEO-optimized meta description under 160 characters"
  }},
  "strategic_insights": {{
    "overall_health": "Assessment of campaign portfolio health",
    "opportunities": ["Market opportunity 1", "Market opportunity 2"],
    "recommendations": ["Strategic recommendation 1", "Strategic recommendation 2"]
  }}
}}

**Important Guidelines:**
- Use actual data from the campaign performance metrics
- Focus on measurable, actionable recommendations
- Generate creative content that aligns with {seo_category}
- Be specific about budget and targeting recommendations
"""

        return prompt

    def generate_d2c_insights(self, campaign_data: pd.DataFrame, seo_category: str) -> Optional[Dict[str, Any]]:
        """
        Generate AI-powered insights from D2C campaign data.

        Args:
            campaign_data: DataFrame with D2C campaign metrics
            seo_category: SEO category for the campaigns

        Returns:
            Dict containing structured D2C insights or None if failed
        """
        logger.info(f"Generating D2C insights for {seo_category} category with {len(campaign_data)} campaigns")

        try:
            # Create the analysis prompt
            prompt = self._create_d2c_analysis_prompt(campaign_data, seo_category)

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            if not response.text:
                logger.error("Empty response from Gemini API")
                return None

            # Parse JSON response
            try:
                insights = json.loads(response.text)
                logger.info(f"Successfully generated D2C insights for {seo_category}")
                return insights

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response.text[:500]}...")
                return None

        except Exception as e:
            logger.error(f"Error generating D2C insights: {e}")
            return None

    def save_insights_to_file(self, insights: Dict[str, Any], filename: str) -> bool:
        """
        Save insights to a JSON file with metadata.

        Args:
            insights: Insights dictionary to save
            filename: Output filename

        Returns:
            bool: True if saved successfully
        """
        try:
            # Add metadata
            insights_with_metadata = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "generator_version": "1.0"
                },
                "insights": insights
            }

            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(insights_with_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Insights saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving insights to file: {e}")
            return False
