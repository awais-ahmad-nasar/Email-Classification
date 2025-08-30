# main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import ValidationError
from models import EmailRequest, EmailResponse
from classifier import PretrainedEmailClassifier
import os
import uvicorn

app = FastAPI(title="Email Classifier API", version="1.0.0")

# Initialize classifier
classifier = PretrainedEmailClassifier(model_type="distilbert")

# Create static directory if it doesn't exist
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Mount static directory for serving HTML
try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
except:
    print(f"Warning: Could not mount static directory '{static_dir}'")


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"message": "Email Classifier API is running", "docs": "/docs"}


@app.post("/api/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    """Classify an email based on subject and body"""
    try:
        result = classifier.classify_email(
            subject=request.subject,
            body=request.body,
            return_probabilities=request.return_probabilities
        )
        return EmailResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return classifier.get_model_info()


@app.get("/api/category-details/{category}")
async def get_category_details(category: str):
    """Get detailed information about a specific category"""
    valid_categories = list(classifier.categories.values())
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {valid_categories}"
        )
    return classifier.get_category_details(category)


@app.post("/api/analyze-detailed")
async def analyze_text_detailed(request: EmailRequest):
    """Provide detailed analysis showing scores for all categories"""
    try:
        analysis = classifier.analyze_text_detailed(
            subject=request.subject,
            body=request.body
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": classifier.classifier is not None}


def run_server():
    """Run the FastAPI server"""
    print("Starting Enhanced Email Classifier Server...")
    print("Server will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    # Test the enhanced classifier first
    print("Testing Enhanced Email Classifier...")

    test_emails = [
        {
            'subject': 'Your Monthly Bank Statement is Ready',
            'body': 'Dear customer, your account statement for January 2024 is now available online. Please log in to your account to view your balance and transactions.'
        },
        {
            'subject': 'MEGA SALE: 50% OFF Everything!',
            'body': 'Limited time offer! Get 50% discount on all products. Use code SAVE50 at checkout. Free shipping on orders over $50. Shop now before this exclusive deal expires!'
        },
        {
            'subject': 'Hey, want to grab coffee this weekend?',
            'body': 'Hi there! It has been a while since we caught up. Would you like to grab coffee this Saturday afternoon? I miss our conversations and would love to hear how you\'re doing.'
        },
        {
            'subject': 'URGENT: You have won $1,000,000!',
            'body': 'Congratulations! You are our lucky winner of the international lottery. You have won $1,000,000! Send your bank details immediately to claim your prize. Act now before this opportunity expires!'
        },
        {
            'subject': 'Your Amazon Order Has Shipped',
            'body': 'Your order #123-4567890-1234567 has been shipped via UPS and will arrive by Thursday. Track your package using the tracking number: 1Z999AA1234567890. Thank you for your purchase!'
        },
        {
            'subject': 'Flight Confirmation - NYC to LAX',
            'body': 'Your flight is confirmed! Flight AA123 from New York (JFK) to Los Angeles (LAX) on March 15, 2024. Departure: 8:00 AM, Arrival: 11:30 AM PST. Check-in opens 24 hours before departure.'
        },
        {
            'subject': 'New Reply to Your Forum Post',
            'body': 'Hello forum member! User @TechExpert has replied to your post "Best Programming Languages 2024" in the Technology Discussion forum. Click here to view the reply and continue the discussion.'
        },
        {
            'subject': 'You were tagged in a Facebook post',
            'body': 'John Smith tagged you in a post: "Great time at the beach with friends!" Click to see the post and add your comment. You have 12 other notifications waiting.'
        }
    ]

    print(f"\nTesting with {len(test_emails)} sample emails:")
    print("=" * 80)

    for i, email in enumerate(test_emails, 1):
        print(f"\nEmail {i}:")
        print(f"Subject: {email['subject']}")
        print(f"Body: {email['body'][:100]}...")

        result = classifier.classify_email(
            email['subject'],
            email['body'],
            return_probabilities=True
        )

        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Method: {result['method']}")

        if 'all_probabilities' in result:
            print("Top 3 predictions:")
            sorted_probs = sorted(result['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True)[:3]
            for category, prob in sorted_probs:
                print(f"  {category}: {prob:.2%}")

    print("\n" + "=" * 80)
    print("Enhanced classification testing complete!")

    # Show model information
    model_info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"- Model Type: {model_info['model_type']}")
    print(f"- Device: {model_info['device']}")
    print(f"- Total Keywords: {model_info['total_keywords']}")
    print(f"- Categories: {len(model_info['categories'])}")
    print(f"- GPU Available: {model_info['gpu_available']}")

    # Show category details
    print(f"\nCategory Keyword Counts:")
    for category in classifier.categories.values():
        details = classifier.get_category_details(category)
        print(f"- {category}: {details['total_keywords']} keywords across {len(details['keyword_groups'])} groups")

    print("\nStarting web server...")

    # Start the server
    run_server()