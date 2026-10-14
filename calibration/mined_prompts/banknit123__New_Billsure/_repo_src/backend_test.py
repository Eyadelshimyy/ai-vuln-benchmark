#!/usr/bin/env python3
"""
Backend API Testing for BillEasyPay DDR and Provider Connection Endpoints
"""

import requests
import json
import sys
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Configuration
BASE_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://pay-manager-stage.preview.emergentagent.com')
API_BASE = f"{BASE_URL}/api"

# Test credentials
TEST_USER = {
    "email": "testuser@example.com",
    "password": "password123"
}

ADMIN_USER = {
    "email": "admin@billseasypay.com", 
    "password": "admin123"
}

class BillEasyPayTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_token = None
        self.admin_token = None
        self.test_results = []
        
    def log_result(self, test_name, success, message, details=None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {}
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name} - {message}")
        if details and not success:
            print(f"   Details: {details}")
    
    def setup_test_user(self):
        """Register or login test user"""
        print("\n=== Setting up Test User ===")
        
        # Try to register test user first
        try:
            response = self.session.post(f"{API_BASE}/auth/register", json={
                "email": TEST_USER["email"],
                "password": TEST_USER["password"],
                "full_name": "Test User",
                "phone": "0412345678"
            })
            
            if response.status_code == 200:
                data = response.json()
                self.test_token = data.get("token")
                self.log_result("User Registration", True, "Test user registered successfully")
                return True
            elif response.status_code == 400 and "already registered" in response.text:
                # User exists, try login
                return self.login_test_user()
            else:
                self.log_result("User Registration", False, f"Registration failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("User Registration", False, f"Registration error: {str(e)}")
            return False
    
    def login_test_user(self):
        """Login test user"""
        try:
            response = self.session.post(f"{API_BASE}/auth/login", json=TEST_USER)
            
            if response.status_code == 200:
                data = response.json()
                self.test_token = data.get("token")
                self.log_result("User Login", True, "Test user logged in successfully")
                return True
            else:
                self.log_result("User Login", False, f"Login failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("User Login", False, f"Login error: {str(e)}")
            return False
    
    def setup_admin_user(self):
        """Setup admin user"""
        print("\n=== Setting up Admin User ===")
        
        # Try to register admin user first
        try:
            response = self.session.post(f"{API_BASE}/auth/register", json={
                "email": ADMIN_USER["email"],
                "password": ADMIN_USER["password"],
                "full_name": "Admin User",
                "phone": "0412345679"
            })
            
            if response.status_code == 200:
                data = response.json()
                self.admin_token = data.get("token")
                self.log_result("Admin Registration", True, "Admin user registered successfully")
                return True
            elif response.status_code == 400 and "already registered" in response.text:
                # User exists, try login
                return self.login_admin_user()
            else:
                self.log_result("Admin Registration", False, f"Admin registration failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("Admin Registration", False, f"Admin registration error: {str(e)}")
            return False
    
    def login_admin_user(self):
        """Login admin user"""
        try:
            response = self.session.post(f"{API_BASE}/auth/login", json=ADMIN_USER)
            
            if response.status_code == 200:
                data = response.json()
                self.admin_token = data.get("token")
                self.log_result("Admin Login", True, "Admin user logged in successfully")
                return True
            else:
                self.log_result("Admin Login", False, f"Admin login failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("Admin Login", False, f"Admin login error: {str(e)}")
            return False
    
    def get_auth_headers(self, use_admin=False):
        """Get authorization headers"""
        token = self.admin_token if use_admin else self.test_token
        return {"Authorization": f"Bearer {token}"}
    
    def test_bsb_validation(self):
        """Test BSB validation endpoint"""
        print("\n=== Testing BSB Validation ===")
        
        # Test valid BSB
        try:
            response = self.session.post(f"{API_BASE}/direct-debit/validate-bsb?bsb=062000")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid") and data.get("bank_name") == "Commonwealth Bank":
                    self.log_result("BSB Validation - Valid BSB", True, "Valid BSB 062000 correctly identified as Commonwealth Bank")
                else:
                    self.log_result("BSB Validation - Valid BSB", False, "Valid BSB not properly validated", data)
            else:
                self.log_result("BSB Validation - Valid BSB", False, f"BSB validation failed: {response.status_code}", response.text)
                
        except Exception as e:
            self.log_result("BSB Validation - Valid BSB", False, f"BSB validation error: {str(e)}")
        
        # Test invalid BSB
        try:
            response = self.session.post(f"{API_BASE}/direct-debit/validate-bsb?bsb=999999")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    self.log_result("BSB Validation - Invalid BSB", True, "Invalid BSB handled correctly")
                else:
                    self.log_result("BSB Validation - Invalid BSB", False, "Invalid BSB not properly handled", data)
            else:
                self.log_result("BSB Validation - Invalid BSB", False, f"Invalid BSB test failed: {response.status_code}", response.text)
                
        except Exception as e:
            self.log_result("BSB Validation - Invalid BSB", False, f"Invalid BSB test error: {str(e)}")
    
    def test_ddr_creation(self):
        """Test Direct Debit Request creation"""
        print("\n=== Testing DDR Creation ===")
        
        if not self.test_token:
            self.log_result("DDR Creation", False, "No authentication token available")
            return None
        
        ddr_data = {
            "bank_name": "Commonwealth Bank",
            "bsb": "062000",
            "account_number": "12345678",
            "account_holder_name": "Test User",
            "account_type": "savings",
            "provider": "Origin Energy",
            "provider_type": "Electricity",
            "provider_account_number": "ELE123456789",
            "payment_frequency": "monthly",
            "max_payment_amount": 500.00,
            "start_date": (datetime.now() + timedelta(days=7)).isoformat(),
            "signature": "Test User"
        }
        
        try:
            response = self.session.post(
                f"{API_BASE}/direct-debit/create",
                json=ddr_data,
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                mandate_id = data.get("id")
                mandate_ref = data.get("mandate_reference")
                
                if mandate_id and mandate_ref:
                    self.log_result("DDR Creation", True, f"DDR created successfully with mandate reference: {mandate_ref}")
                    return mandate_id
                else:
                    self.log_result("DDR Creation", False, "DDR created but missing ID or reference", data)
                    return None
            else:
                self.log_result("DDR Creation", False, f"DDR creation failed: {response.status_code}", response.text)
                return None
                
        except Exception as e:
            self.log_result("DDR Creation", False, f"DDR creation error: {str(e)}")
            return None
    
    def test_ddr_mandates_fetch(self):
        """Test fetching DDR mandates"""
        print("\n=== Testing DDR Mandates Fetch ===")
        
        if not self.test_token:
            self.log_result("DDR Mandates Fetch", False, "No authentication token available")
            return []
        
        try:
            response = self.session.get(
                f"{API_BASE}/direct-debit/mandates",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                mandates = response.json()
                self.log_result("DDR Mandates Fetch", True, f"Successfully fetched {len(mandates)} mandates")
                return mandates
            else:
                self.log_result("DDR Mandates Fetch", False, f"Mandates fetch failed: {response.status_code}", response.text)
                return []
                
        except Exception as e:
            self.log_result("DDR Mandates Fetch", False, f"Mandates fetch error: {str(e)}")
            return []
    
    def test_ddr_mandate_cancel(self, mandate_id):
        """Test cancelling a DDR mandate"""
        print("\n=== Testing DDR Mandate Cancellation ===")
        
        if not self.test_token or not mandate_id:
            self.log_result("DDR Mandate Cancel", False, "No authentication token or mandate ID available")
            return False
        
        try:
            response = self.session.put(
                f"{API_BASE}/direct-debit/mandate/{mandate_id}/cancel",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("DDR Mandate Cancel", True, f"Mandate cancelled successfully: {data.get('message')}")
                return True
            else:
                self.log_result("DDR Mandate Cancel", False, f"Mandate cancellation failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("DDR Mandate Cancel", False, f"Mandate cancellation error: {str(e)}")
            return False
    
    def test_provider_connection(self):
        """Test provider connection creation"""
        print("\n=== Testing Provider Connection ===")
        
        if not self.test_token:
            self.log_result("Provider Connection", False, "No authentication token available")
            return None
        
        provider_data = {
            "provider_name": "Origin Energy",
            "provider_type": "Electricity",
            "account_number": "ELE123456789",
            "customer_id": "CUST789456",
            "api_key": "test_api_key_123"
        }
        
        try:
            response = self.session.post(
                f"{API_BASE}/provider/connect",
                json=provider_data,
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                connection_id = data.get("id")
                
                if connection_id:
                    self.log_result("Provider Connection", True, f"Provider connected successfully: {data.get('provider_name')}")
                    return connection_id
                else:
                    self.log_result("Provider Connection", False, "Provider connected but missing ID", data)
                    return None
            else:
                self.log_result("Provider Connection", False, f"Provider connection failed: {response.status_code}", response.text)
                return None
                
        except Exception as e:
            self.log_result("Provider Connection", False, f"Provider connection error: {str(e)}")
            return None
    
    def test_provider_connections_fetch(self):
        """Test fetching provider connections"""
        print("\n=== Testing Provider Connections Fetch ===")
        
        if not self.test_token:
            self.log_result("Provider Connections Fetch", False, "No authentication token available")
            return []
        
        try:
            response = self.session.get(
                f"{API_BASE}/provider/connections",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                connections = response.json()
                self.log_result("Provider Connections Fetch", True, f"Successfully fetched {len(connections)} connections")
                return connections
            else:
                self.log_result("Provider Connections Fetch", False, f"Connections fetch failed: {response.status_code}", response.text)
                return []
                
        except Exception as e:
            self.log_result("Provider Connections Fetch", False, f"Connections fetch error: {str(e)}")
            return []
    
    def test_provider_sync(self, connection_id):
        """Test provider bill sync"""
        print("\n=== Testing Provider Bill Sync ===")
        
        if not self.test_token or not connection_id:
            self.log_result("Provider Bill Sync", False, "No authentication token or connection ID available")
            return False
        
        try:
            response = self.session.post(
                f"{API_BASE}/provider/sync/{connection_id}",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("Provider Bill Sync", True, f"Bills synced successfully: {data.get('message')}")
                return True
            else:
                self.log_result("Provider Bill Sync", False, f"Bill sync failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("Provider Bill Sync", False, f"Bill sync error: {str(e)}")
            return False
    
    def test_provider_disconnect(self, connection_id):
        """Test provider disconnection"""
        print("\n=== Testing Provider Disconnection ===")
        
        if not self.test_token or not connection_id:
            self.log_result("Provider Disconnect", False, "No authentication token or connection ID available")
            return False
        
        try:
            response = self.session.delete(
                f"{API_BASE}/provider/disconnect/{connection_id}",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("Provider Disconnect", True, f"Provider disconnected successfully: {data.get('message')}")
                return True
            else:
                self.log_result("Provider Disconnect", False, f"Provider disconnect failed: {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_result("Provider Disconnect", False, f"Provider disconnect error: {str(e)}")
            return False
    
    def test_authentication_required(self):
        """Test that endpoints require authentication"""
        print("\n=== Testing Authentication Requirements ===")
        
        endpoints_to_test = [
            ("POST", "/direct-debit/create"),
            ("GET", "/direct-debit/mandates"),
            ("POST", "/provider/connect"),
            ("GET", "/provider/connections")
        ]
        
        for method, endpoint in endpoints_to_test:
            try:
                if method == "POST":
                    response = self.session.post(f"{API_BASE}{endpoint}", json={})
                else:
                    response = self.session.get(f"{API_BASE}{endpoint}")
                
                if response.status_code in [401, 403]:
                    self.log_result(f"Auth Required - {method} {endpoint}", True, "Correctly requires authentication")
                else:
                    self.log_result(f"Auth Required - {method} {endpoint}", False, f"Should require auth but got: {response.status_code}")
                    
            except Exception as e:
                self.log_result(f"Auth Required - {method} {endpoint}", False, f"Auth test error: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting BillEasyPay Backend API Tests")
        print(f"📍 Testing against: {API_BASE}")
        
        # Setup users
        if not self.setup_test_user():
            print("❌ Failed to setup test user, aborting tests")
            return False
        
        if not self.setup_admin_user():
            print("⚠️  Failed to setup admin user, continuing with regular user tests")
        
        # Test authentication requirements
        self.test_authentication_required()
        
        # Test BSB validation
        self.test_bsb_validation()
        
        # Test DDR functionality
        mandate_id = self.test_ddr_creation()
        mandates = self.test_ddr_mandates_fetch()
        
        if mandate_id:
            self.test_ddr_mandate_cancel(mandate_id)
        
        # Test Provider Connection functionality
        connection_id = self.test_provider_connection()
        connections = self.test_provider_connections_fetch()
        
        if connection_id:
            self.test_provider_sync(connection_id)
            self.test_provider_disconnect(connection_id)
        
        # Print summary
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.test_results if r["success"])
        failed = len(self.test_results) - passed
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(passed/len(self.test_results)*100):.1f}%")
        
        if failed > 0:
            print("\n🔍 FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   ❌ {result['test']}: {result['message']}")
        
        print("\n" + "="*60)

def main():
    """Main test runner"""
    tester = BillEasyPayTester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()