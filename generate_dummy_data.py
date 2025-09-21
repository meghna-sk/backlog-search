import os
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import itertools

# Ensure required directories exist
os.makedirs("rawData", exist_ok=True)

# Expanded base components for generating unique content
BASE_COMPONENTS = {
    "actions": [
        "crash", "fail", "error", "timeout", "hang", "freeze", "corrupt", "leak", "overflow", "underflow",
        "validate", "authenticate", "authorize", "encrypt", "decrypt", "compress", "decompress", "sync", "backup", "restore",
        "render", "display", "update", "refresh", "reload", "cache", "optimize", "scale", "balance", "distribute",
        "process", "queue", "schedule", "execute", "monitor", "log", "track", "analyze", "report", "export",
        "import", "migrate", "transform", "convert", "parse", "serialize", "deserialize", "validate", "sanitize", "filter"
    ],
    "subjects": [
        "application", "system", "database", "server", "client", "service", "module", "component", "interface", "API",
        "user", "session", "token", "cookie", "cache", "memory", "disk", "network", "connection", "socket",
        "file", "document", "image", "video", "audio", "data", "record", "table", "query", "transaction",
        "request", "response", "message", "notification", "email", "SMS", "push", "webhook", "event", "trigger",
        "form", "field", "input", "output", "button", "link", "menu", "navigation", "layout", "theme"
    ],
    "contexts": [
        "startup", "shutdown", "login", "logout", "registration", "authentication", "authorization", "permission", "access", "security",
        "performance", "scalability", "reliability", "availability", "maintainability", "usability", "compatibility", "integration", "deployment", "configuration",
        "mobile", "desktop", "web", "tablet", "responsive", "adaptive", "cross-platform", "browser", "device", "screen",
        "production", "staging", "development", "testing", "debugging", "monitoring", "logging", "analytics", "reporting", "dashboard",
        "backup", "recovery", "disaster", "failover", "redundancy", "clustering", "load-balancing", "caching", "optimization", "tuning"
    ],
    "severities": [
        "critical", "high", "medium", "low", "urgent", "blocking", "major", "minor", "trivial", "cosmetic"
    ],
    "areas": [
        "User Interface", "Backend Services", "Database", "Security", "Performance", "Integration", 
        "Mobile", "Analytics", "Testing", "DevOps", "API", "Authentication", "Authorization", "Data Processing"
    ],
    "applications": [
        "Web Portal", "Mobile App", "Desktop Client", "API Service", "Admin Panel", "Reporting Tool", 
        "Analytics Dashboard", "Customer Portal", "Employee Portal", "Management System"
    ],
    "teams": [
        "Frontend Team", "Backend Team", "DevOps Team", "QA Team", "Product Team", "Design Team", 
        "Security Team", "Data Team", "Mobile Team", "Integration Team"
    ]
}

# Template patterns for different document types
PATTERNS = {
    "bug_reports": {
        "titles": [
            "{action} in {subject} during {context}",
            "{subject} {action}s when {context}",
            "{action} error in {subject} {context}",
            "{subject} fails to {action} during {context}",
            "{context} causes {subject} to {action}",
            "{action} occurs in {subject} {context}",
            "{subject} {action}s unexpectedly during {context}",
            "{context} triggers {action} in {subject}",
            "{subject} {action}s after {context}",
            "{action} prevents {subject} from {context}"
        ],
        "descriptions": [
            "The {subject} {action}s when {context} occurs, causing {severity} impact on system functionality.",
            "During {context}, the {subject} encounters an {action} that prevents normal operation.",
            "A {action} in the {subject} has been identified during {context} testing.",
            "The {subject} {action}s unexpectedly when {context} conditions are met.",
            "When {context} is triggered, the {subject} fails to {action} properly.",
            "The {subject} {action}s due to {context} configuration issues.",
            "During {context}, users experience {action} in the {subject} functionality.",
            "The {subject} {action}s intermittently during {context} operations.",
            "A {action} has been reported in the {subject} when {context} is active.",
            "The {subject} {action}s consistently during {context} scenarios."
        ],
        "causes": [
            "The {action} is caused by {context} configuration in the {subject}.",
            "{context} logic in the {subject} leads to {action} behavior.",
            "The {subject} {action}s due to {context} implementation issues.",
            "{context} handling in the {subject} triggers the {action}.",
            "The {action} occurs because of {context} limitations in the {subject}.",
            "The {subject} {action}s when {context} conditions are not properly handled.",
            "{context} processing in the {subject} causes the {action} to occur.",
            "The {action} is triggered by {context} state changes in the {subject}.",
            "The {subject} {action}s due to {context} resource constraints.",
            "{context} validation in the {subject} fails, causing the {action}."
        ],
        "solutions": [
            "Implement proper {context} handling in the {subject} to prevent {action}.",
            "Add {context} validation to the {subject} to avoid {action} scenarios.",
            "Modify the {subject} to handle {context} conditions without {action}.",
            "Update {context} logic in the {subject} to resolve the {action}.",
            "Implement {context} error handling in the {subject} to prevent {action}.",
            "Add {context} checks to the {subject} before {action} can occur.",
            "Modify the {subject} to gracefully handle {context} without {action}.",
            "Implement {context} retry logic in the {subject} to avoid {action}.",
            "Add {context} monitoring to the {subject} to detect {action} early.",
            "Update the {subject} to properly manage {context} resources."
        ],
        "verifications": [
            "Verify that the {subject} no longer {action}s during {context}.",
            "Test {context} scenarios to ensure the {subject} handles them properly.",
            "Confirm that {action} does not occur in the {subject} when {context} is active.",
            "Validate that the {subject} {action}s correctly during {context} testing.",
            "Check that {context} operations in the {subject} work without {action}.",
            "Ensure the {subject} properly handles {context} without {action}.",
            "Verify {context} functionality in the {subject} after the fix.",
            "Test that the {subject} {action}s appropriately during {context}.",
            "Confirm {context} behavior in the {subject} is now correct.",
            "Validate that the {subject} {action}s as expected during {context}."
        ]
    },
    "feature_requests": {
        "titles": [
            "Add {action} functionality to {subject}",
            "Implement {context} support in {subject}",
            "Create {action} feature for {subject}",
            "Add {context} capabilities to {subject}",
            "Implement {action} in {subject} {context}",
            "Add {subject} {action} for {context}",
            "Create {context} {action} in {subject}",
            "Implement {subject} {action} functionality",
            "Add {action} support for {subject} {context}",
            "Create {context} {action} feature"
        ],
        "descriptions": [
            "Users need {action} functionality in the {subject} to improve {context} experience.",
            "Implementing {action} in the {subject} would enhance {context} capabilities.",
            "Adding {action} support to the {subject} would benefit {context} operations.",
            "The {subject} should include {action} features for better {context} handling.",
            "Users request {action} functionality in the {subject} for {context} purposes.",
            "Implementing {action} in the {subject} would improve {context} efficiency.",
            "Adding {action} to the {subject} would enhance {context} user experience.",
            "The {subject} needs {action} capabilities for {context} requirements.",
            "Users would benefit from {action} functionality in the {subject} for {context}.",
            "Implementing {action} in the {subject} would support {context} workflows."
        ]
    },
    "requirements": {
        "titles": [
            "{subject} must support {action} for {context}",
            "Implement {action} requirements in {subject}",
            "{subject} {action} specifications for {context}",
            "Define {action} standards for {subject}",
            "{subject} {action} compliance for {context}",
            "Establish {action} guidelines for {subject}",
            "{subject} {action} requirements for {context}",
            "Define {action} protocols for {subject}",
            "{subject} {action} standards for {context}",
            "Implement {action} policies for {subject}"
        ],
        "descriptions": [
            "The {subject} must implement {action} to meet {context} requirements.",
            "System {action} capabilities are required for {subject} {context} compliance.",
            "The {subject} needs {action} functionality to satisfy {context} standards.",
            "Implementing {action} in the {subject} is necessary for {context} requirements.",
            "The {subject} must support {action} to ensure {context} compliance.",
            "System {action} features are essential for {subject} {context} operations.",
            "The {subject} requires {action} implementation for {context} functionality.",
            "Implementing {action} in the {subject} supports {context} requirements.",
            "The {subject} must provide {action} capabilities for {context} needs.",
            "System {action} functionality is required for {subject} {context} support."
        ]
    }
}

def generate_unique_content(pattern_type, record_id):
    """Generate unique content using patterns and base components"""
    patterns = PATTERNS[pattern_type]
    
    # Use record_id to ensure uniqueness
    random.seed(hash(record_id))
    
    # Generate unique combinations
    action = random.choice(BASE_COMPONENTS["actions"])
    subject = random.choice(BASE_COMPONENTS["subjects"])
    context = random.choice(BASE_COMPONENTS["contexts"])
    severity = random.choice(BASE_COMPONENTS["severities"])
    
    # Add more variation using record_id
    variation_id = int(record_id.split('-')[1], 16) % 1000
    
    # Generate title
    title_pattern = random.choice(patterns["titles"])
    title = title_pattern.format(action=action, subject=subject, context=context, severity=severity)
    
    # Add unique identifier to title
    title += f" (ID: {variation_id})"
    
    # Generate description
    desc_pattern = random.choice(patterns["descriptions"])
    description = desc_pattern.format(action=action, subject=subject, context=context, severity=severity)
    
    # Add unique details
    description += f" This issue was identified in environment {variation_id % 10 + 1} and affects version {variation_id % 5 + 1}.{variation_id % 10}."
    
    # Generate other fields based on pattern type
    if pattern_type == "bug_reports":
        cause_pattern = random.choice(patterns["causes"])
        cause = cause_pattern.format(action=action, subject=subject, context=context, severity=severity)
        
        solution_pattern = random.choice(patterns["solutions"])
        solution = solution_pattern.format(action=action, subject=subject, context=context, severity=severity)
        
        verification_pattern = random.choice(patterns["verifications"])
        verification = verification_pattern.format(action=action, subject=subject, context=context, severity=severity)
        
        return {
            "title": title,
            "description": description,
            "cause": cause,
            "solution": solution,
            "verification": verification
        }
    else:
        return {
            "title": title,
            "description": description
        }

def generate_unique_tags(document_type, title, description, record_id):
    """Generate unique tags based on content and record ID"""
    random.seed(hash(record_id))
    
    tag_categories = {
        "bug": ["critical", "high", "medium", "low", "ui", "backend", "database", "performance", "security", "mobile", "api", "integration"],
        "feature": ["enhancement", "ui", "backend", "mobile", "integration", "security", "performance", "api", "database", "analytics"],
        "requirement": ["functional", "non-functional", "security", "performance", "compliance", "integration", "scalability", "usability"]
    }
    
    tags = []
    if document_type == "bug_reports":
        tags.extend(random.sample(tag_categories["bug"], random.randint(3, 6)))
    elif document_type == "feature_requests":
        tags.extend(random.sample(tag_categories["feature"], random.randint(3, 6)))
    else:
        tags.extend(random.sample(tag_categories["requirement"], random.randint(3, 6)))
    
    # Add content-based tags
    content_lower = (title + " " + description).lower()
    
    if any(word in content_lower for word in ["crash", "fail", "error", "critical"]):
        if "critical" not in tags:
            tags.append("critical")
    
    if any(word in content_lower for word in ["performance", "slow", "timeout", "memory"]):
        if "performance" not in tags:
            tags.append("performance")
    
    if any(word in content_lower for word in ["database", "data", "query", "sql"]):
        if "database" not in tags:
            tags.append("database")
    
    if any(word in content_lower for word in ["auth", "login", "security", "permission"]):
        if "security" not in tags:
            tags.append("security")
    
    if any(word in content_lower for word in ["ui", "interface", "button", "form", "display"]):
        if "ui" not in tags:
            tags.append("ui")
    
    if any(word in content_lower for word in ["mobile", "app", "device", "responsive"]):
        if "mobile" not in tags:
            tags.append("mobile")
    
    # Add unique tag based on record ID
    unique_tag = f"issue-{int(record_id.split('-')[1], 16) % 1000}"
    tags.append(unique_tag)
    
    return ", ".join(list(set(tags)))  # Remove duplicates

def generate_dummy_data(num_records=10000):
    """Generate large amounts of unique dummy data"""
    
    print(f"Generating {num_records} unique dummy records...")
    
    records = []
    used_ids = set()
    
    for i in range(num_records):
        # Generate unique ID
        while True:
            record_id = f"DOC-{str(uuid.uuid4())[:8].upper()}"
            if record_id not in used_ids:
                used_ids.add(record_id)
                break
        
        # Choose document type
        doc_type = random.choice(["bug_reports", "feature_requests", "requirements"])
        
        # Generate unique content
        content = generate_unique_content(doc_type, record_id)
        
        # Generate unique tags
        tags = generate_unique_tags(doc_type, content["title"], content["description"], record_id)
        
        # Smart area assignment based on content
        area = random.choice(BASE_COMPONENTS["areas"])
        content_lower = (content["title"] + " " + content["description"]).lower()
        
        if any(word in content_lower for word in ["ui", "interface", "button", "form", "display"]):
            area = "User Interface"
        elif any(word in content_lower for word in ["database", "data", "query", "sql"]):
            area = "Database"
        elif any(word in content_lower for word in ["auth", "login", "security", "permission"]):
            area = "Security"
        elif any(word in content_lower for word in ["performance", "slow", "timeout", "memory"]):
            area = "Performance"
        elif any(word in content_lower for word in ["mobile", "app", "device", "responsive"]):
            area = "Mobile"
        elif any(word in content_lower for word in ["api", "integration", "service", "endpoint"]):
            area = "Integration"
        
        # Create record
        record = {
            "ID": record_id,
            "Name": content["title"],
            "Description": content["description"],
            "Tags": tags,
            "Issue Key": f"KEY-{random.randint(10000, 99999)}",
            "Area": area,
            "Application": random.choice(BASE_COMPONENTS["applications"]),
            "Teams": random.choice(BASE_COMPONENTS["teams"])
        }
        
        # Add type-specific fields
        if doc_type == "bug_reports":
            record.update({
                "Cause": content["cause"],
                "Solution": content["solution"],
                "Verification": content["verification"],
                "Deferral Justification": random.choice([
                    "Low priority", "Resource constraints", "Technical complexity", 
                    "Dependency on other features", "User impact minimal", "Requires additional research",
                    "Not in current roadmap", "Infrastructure dependency"
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
                    "Dependency on infrastructure changes", "User demand insufficient",
                    "Technical complexity too high", "Resource allocation needed"
                ]) if random.random() < 0.4 else None,
                "Rationale": random.choice([
                    "User experience improvement", "Business value", "Competitive advantage",
                    "Technical debt reduction", "Compliance requirement", "Performance enhancement",
                    "Security improvement", "Integration capability"
                ])
            })
        else:  # requirements
            record.update({
                "Cause": None,
                "Solution": None,
                "Verification": None,
                "Deferral Justification": None,
                "Rationale": random.choice([
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
                ])
            })
        
        records.append(record)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} unique records...")
    
    return records

def create_csv_files(records):
    """Create multiple CSV files to simulate different data sources"""
    
    # Split records into different files
    bug_reports = [r for r in records if r.get("Cause") is not None]
    feature_requests = [r for r in records if r.get("Rationale") is not None and r.get("Cause") is None]
    incidents = [r for r in records if r.get("Cause") is not None and r.get("Rationale") is None]
    
    # Create different CSV files that match what model.py expects
    files_to_create = [
        ("bug_reports.csv", bug_reports[:len(bug_reports)//2]),
        ("feature_requests.csv", feature_requests[:len(feature_requests)//2]),
        ("incidents.csv", incidents[:len(incidents)//2]),
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
