import os
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np

# Ensure required directories exist
os.makedirs("rawData", exist_ok=True)

# Sample data templates for different types of documents
DOCUMENT_TYPES = {
    "bug_reports": {
        "titles": [
            "Application crashes on startup", "Data validation error in form submission", "Memory leak in background process",
            "UI elements not rendering correctly", "Database connection timeout", "File upload fails for large files",
            "Authentication token expires too quickly", "Search functionality returns no results", "Email notifications not being sent",
            "Performance degradation under load", "Button click not registering", "Form submission hangs indefinitely",
            "Data not saving to database", "User session expires unexpectedly", "Mobile app crashes on iOS",
            "API endpoint returns 500 error", "Login fails with valid credentials", "Dashboard not loading data",
            "Export function generates empty file", "Calendar events not syncing", "Push notifications not received",
            "Video playback stuttering", "Image upload corrupted", "Search results not sorted correctly",
            "User profile picture not updating", "Password reset email not sent", "Two-factor authentication failing",
            "Dark mode not applying to all screens", "Keyboard shortcuts not working", "Drag and drop functionality broken",
            "Print preview shows blank page", "Backup process fails silently", "System logs showing errors",
            "Cache not clearing properly", "WebSocket connection drops", "File download incomplete",
            "Report generation takes too long", "User permissions not enforced", "Data export missing columns",
            "Mobile responsive layout broken", "Browser compatibility issues", "SSL certificate errors",
            "Database query timeout", "Memory usage spikes", "CPU usage at 100%", "Disk space full error",
            "Network request timeout", "Invalid JSON response", "CORS policy blocking requests",
            "Session cookie not set", "CSRF token validation failed", "Rate limiting too aggressive",
            "Third-party API integration broken", "Payment processing error", "Email template rendering issue",
            "PDF generation corrupted", "Excel export formatting wrong", "CSV import validation failing",
            "User interface freezing", "Loading spinner never stops", "Error messages not user-friendly",
            "Accessibility features not working", "Screen reader compatibility issues", "Color contrast problems",
            "Font rendering issues", "Layout shifts on page load", "JavaScript errors in console",
            "CSS styles not applying", "Responsive breakpoints incorrect", "Mobile navigation not working",
            "Desktop notifications not showing", "Offline mode not functioning", "Sync conflicts not resolved",
            "Data integrity check failed", "Audit trail missing entries", "Version control conflicts",
            "Deployment pipeline broken", "Environment variables not set", "Configuration file corrupted",
            "Service discovery failing", "Load balancer not distributing", "CDN not serving content",
            "Monitoring alerts not firing", "Log aggregation not working", "Metrics collection stopped",
            "Health check endpoint down", "Database migration failed", "Schema validation errors",
            "API versioning conflicts", "Backward compatibility broken", "Feature flag not working",
            "A/B test not running", "Analytics tracking missing", "User behavior not recorded",
            "Performance metrics degraded", "Security scan failed", "Vulnerability detected",
            "Penetration test issues", "Compliance check failed", "Audit findings not addressed"
        ],
        "descriptions": [
            "The application crashes immediately upon startup with error code 0x0000005",
            "Form validation fails to catch invalid email addresses in certain formats",
            "Memory usage increases continuously during long-running operations",
            "UI components appear misaligned on different screen resolutions",
            "Database queries timeout after 30 seconds causing application errors",
            "File uploads larger than 10MB fail with network error",
            "Authentication tokens expire after 15 minutes instead of configured 1 hour",
            "Search queries return empty results even when matching data exists",
            "Email notifications are queued but never actually sent to users",
            "Application response time increases significantly under concurrent load"
        ],
        "causes": [
            "Null pointer exception in initialization code",
            "Regex pattern doesn't account for international email formats",
            "Objects not properly garbage collected",
            "CSS media queries not handling all breakpoints",
            "Database connection pool exhausted",
            "Server timeout configuration too restrictive",
            "Token expiration logic has incorrect time calculation",
            "Search index not properly synchronized with database",
            "SMTP server configuration incorrect",
            "Database queries not optimized for concurrent access"
        ],
        "solutions": [
            "Add null checks before accessing object properties",
            "Update email validation regex to support international formats",
            "Implement proper object disposal and memory management",
            "Add additional CSS breakpoints for better responsive design",
            "Increase database connection pool size and add monitoring",
            "Adjust server timeout settings and add retry logic",
            "Fix token expiration calculation to use correct timezone",
            "Implement real-time search index synchronization",
            "Update SMTP configuration with correct server details",
            "Optimize database queries and add proper indexing"
        ],
        "verifications": [
            "Application starts successfully on multiple test environments",
            "Email validation accepts valid international formats",
            "Memory usage remains stable during extended testing",
            "UI renders correctly across different screen sizes",
            "Database operations complete within acceptable time limits",
            "Large files upload successfully without errors",
            "Authentication tokens persist for configured duration",
            "Search returns accurate results for various query types",
            "Email notifications are delivered to test recipients",
            "Application maintains performance under stress testing"
        ]
    },
    "feature_requests": {
        "titles": [
            "Add dark mode theme support",
            "Implement user role management",
            "Add export functionality for reports",
            "Create mobile app version",
            "Add real-time notifications",
            "Implement advanced search filters",
            "Add data visualization charts",
            "Create API for third-party integration",
            "Add multi-language support",
            "Implement automated testing framework"
        ],
        "descriptions": [
            "Users have requested a dark mode theme to reduce eye strain during night usage",
            "Need comprehensive role-based access control for different user types",
            "Users want to export reports in various formats (PDF, Excel, CSV)",
            "Mobile application needed for field workers and remote access",
            "Real-time push notifications for important system events",
            "Advanced filtering options for better data discovery",
            "Interactive charts and graphs for data analysis",
            "RESTful API for integration with external systems",
            "Support for multiple languages to serve international users",
            "Automated testing to ensure code quality and reduce bugs"
        ],
        "areas": [
            "User Interface", "Backend Services", "Database", "Security", 
            "Performance", "Integration", "Mobile", "Analytics", "Testing"
        ],
        "applications": [
            "Web Portal", "Mobile App", "Desktop Client", "API Service", 
            "Admin Panel", "Reporting Tool", "Analytics Dashboard"
        ],
        "teams": [
            "Frontend Team", "Backend Team", "DevOps Team", "QA Team", 
            "Product Team", "Design Team", "Security Team"
        ]
    },
    "requirements": {
        "titles": [
            "User authentication system requirements",
            "Data backup and recovery specifications",
            "Performance benchmarks for API endpoints",
            "Security compliance requirements",
            "Scalability requirements for user growth",
            "Integration requirements with external systems",
            "Accessibility compliance standards",
            "Data privacy and GDPR compliance",
            "Monitoring and logging requirements",
            "Disaster recovery procedures"
        ],
        "descriptions": [
            "System must support multi-factor authentication and password policies",
            "Automated daily backups with point-in-time recovery capabilities",
            "API response times must not exceed 200ms for 95% of requests",
            "System must comply with SOC 2 Type II security standards",
            "Architecture must support 10x current user load without degradation",
            "Integration with CRM, ERP, and payment processing systems",
            "WCAG 2.1 AA compliance for all user interfaces",
            "Full GDPR compliance with data portability and deletion rights",
            "Comprehensive monitoring with alerting for critical system metrics",
            "RTO of 4 hours and RPO of 1 hour for disaster recovery"
        ],
        "rationales": [
            "Security best practices require strong authentication mechanisms",
            "Data loss prevention is critical for business continuity",
            "User experience depends on responsive system performance",
            "Regulatory compliance is mandatory for business operations",
            "Future growth requires scalable architecture design",
            "Business processes require seamless system integration",
            "Accessibility ensures inclusive user experience",
            "Legal requirements mandate data protection compliance",
            "Operational excellence requires comprehensive monitoring",
            "Business continuity requires robust disaster recovery"
        ]
    }
}

def generate_random_text(template_list, min_words=5, max_words=20):
    """Generate random text by combining templates"""
    num_sentences = random.randint(1, 3)
    sentences = []
    
    for _ in range(num_sentences):
        base_text = random.choice(template_list)
        # Add some variation
        if random.random() < 0.3:
            base_text += f" Additional details: {random.choice(template_list).lower()}"
        sentences.append(base_text)
    
    return ". ".join(sentences)

def generate_tags(document_type, title, description):
    """Generate relevant tags based on content"""
    tag_categories = {
        "bug": ["critical", "high", "medium", "low", "ui", "backend", "database", "performance"],
        "feature": ["enhancement", "ui", "backend", "mobile", "integration", "security"],
        "requirement": ["functional", "non-functional", "security", "performance", "compliance"]
    }
    
    tags = []
    if document_type == "bug_reports":
        tags.extend(random.sample(tag_categories["bug"], random.randint(2, 4)))
    elif document_type == "feature_requests":
        tags.extend(random.sample(tag_categories["feature"], random.randint(2, 4)))
    else:
        tags.extend(random.sample(tag_categories["requirement"], random.randint(2, 4)))
    
    # Ensure we have some guaranteed matches for common searches
    if "crash" in title.lower() or "crash" in description.lower():
        if "critical" not in tags:
            tags.append("critical")
        if "ui" not in tags and "backend" not in tags:
            tags.append(random.choice(["ui", "backend"]))
    
    if "performance" in title.lower() or "performance" in description.lower():
        if "performance" not in tags:
            tags.append("performance")
        if "high" not in tags:
            tags.append("high")
    
    if "database" in title.lower() or "database" in description.lower():
        if "database" not in tags:
            tags.append("database")
        if "critical" not in tags:
            tags.append("critical")
    
    if "authentication" in title.lower() or "login" in title.lower():
        if "security" not in tags:
            tags.append("security")
        if "critical" not in tags:
            tags.append("critical")
    
    return ", ".join(tags)

def generate_dummy_data(num_records=10000):
    """Generate large amounts of dummy data"""
    
    print(f"Generating {num_records} dummy records...")
    
    records = []
    
    for i in range(num_records):
        # Choose document type
        doc_type = random.choice(list(DOCUMENT_TYPES.keys()))
        templates = DOCUMENT_TYPES[doc_type]
        
        # Generate unique ID
        record_id = f"DOC-{str(uuid.uuid4())[:8].upper()}"
        
        # Generate title and description
        title = random.choice(templates["titles"])
        description = random.choice(templates["descriptions"])
        
        # Add some variation to avoid exact duplicates
        if random.random() < 0.2:
            title = f"{title} (Variant {random.randint(1, 5)})"
        if random.random() < 0.3:
            description = f"{description} This issue affects version {random.randint(1, 5)}.{random.randint(0, 9)}."
        
        # Generate tags
        tags = generate_tags(doc_type, title, description)
        
        # Smart area assignment based on content
        area = random.choice(templates.get("areas", ["General"]))
        if "crash" in title.lower() or "ui" in title.lower() or "interface" in title.lower():
            area = random.choice(["User Interface", "Backend Services"])
        elif "database" in title.lower() or "data" in title.lower():
            area = "Database"
        elif "security" in title.lower() or "auth" in title.lower() or "login" in title.lower():
            area = "Security"
        elif "performance" in title.lower() or "slow" in title.lower():
            area = "Performance"
        elif "mobile" in title.lower() or "app" in title.lower():
            area = "Mobile"
        
        # Create record based on document type
        record = {
            "ID": record_id,
            "Name": title,
            "Description": description,
            "Tags": tags,
            "Issue Key": f"KEY-{random.randint(1000, 9999)}",
            "Area": area,
            "Application": random.choice(templates.get("applications", ["Main Application"])),
            "Teams": random.choice(templates.get("teams", ["Development Team"]))
        }
        
        # Add type-specific fields
        if doc_type == "bug_reports":
            record.update({
                "Cause": random.choice(templates["causes"]),
                "Solution": random.choice(templates["solutions"]),
                "Verification": random.choice(templates["verifications"]),
                "Deferral Justification": random.choice([
                    "Low priority", "Resource constraints", "Technical complexity", 
                    "Dependency on other features", "User impact minimal"
                ]) if random.random() < 0.3 else None,
                "Rationale": None
            })
        elif doc_type == "feature_requests":
            record.update({
                "Cause": None,
                "Solution": None,
                "Verification": None,
                "Deferral Justification": random.choice([
                    "Not in current roadmap", "Requires additional research", 
                    "Dependency on infrastructure changes", "User demand insufficient"
                ]) if random.random() < 0.4 else None,
                "Rationale": random.choice([
                    "User experience improvement", "Business value", "Competitive advantage",
                    "Technical debt reduction", "Compliance requirement"
                ])
            })
        else:  # requirements
            record.update({
                "Cause": None,
                "Solution": None,
                "Verification": None,
                "Deferral Justification": None,
                "Rationale": random.choice(templates["rationales"])
            })
        
        records.append(record)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} records...")
    
    return records

def create_csv_files(records):
    """Create multiple CSV files to simulate different data sources"""
    
    # Split records into different files
    bug_reports = [r for r in records if r.get("Cause") is not None]
    feature_requests = [r for r in records if r.get("Rationale") is not None and r.get("Cause") is None]
    requirements = [r for r in records if r.get("Rationale") is not None and r.get("Cause") is None and r.get("Deferral Justification") is None]
    
    # Create different CSV files
    files_to_create = [
        ("bug_reports.csv", bug_reports[:len(bug_reports)//2]),
        ("feature_requests.csv", feature_requests[:len(feature_requests)//2]),
        ("requirements.csv", requirements),
        ("mixed_documents.csv", records[len(records)//2:])  # Second half as mixed
    ]
    
    for filename, data in files_to_create:
        if data:  # Only create file if there's data
            df = pd.DataFrame(data)
            filepath = os.path.join("rawData", filename)
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"Created {filename} with {len(data)} records")
    
    print(f"\nTotal records generated: {len(records)}")
    print("Files created in rawData/ directory")

def main():
    """Main function to generate dummy data"""
    print("=== Dummy Data Generator ===")
    print("This will generate large amounts of dummy data for training the search model")
    
    # Generate data
    num_records = 25000  # Even faster for testing
    records = generate_dummy_data(num_records)
    
    # Create CSV files
    create_csv_files(records)
    
    print("\n=== Data Generation Complete ===")
    print("You can now run 'python model.py' to process this data and build the search index")

if __name__ == "__main__":
    main()
