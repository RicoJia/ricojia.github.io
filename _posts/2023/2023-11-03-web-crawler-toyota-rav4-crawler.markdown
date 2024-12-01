---
layout: post
title: Web Devel - web-crawler-toyota-rav4-crawler
date: '2023-11-03 13:19'
subtitle: Google Drive
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - Web Devel
---

[Complete web crawler code can be found here](https://github.com/RicoJia/SimpleRoboticsUtils/blob/master/SimpleRoboticsPythonUtils/simple_robotics_python_utils/webcrawlers/car_dealership_crawler.py)

## Beautiful Soup and Selium

Beautiful Soup is a Python library for parsing HTML and XML document. It is able to extract elements by class or ids *in static contents*.

- `soup.find_all()`

```python
.find(): Finds the first match for the given tag or selector.
.find_all(): Finds all matches within the entire subtree.
```

But `requests` will not fetch dynamically loaded elements (such as search results), since it only retrieves the initial HTML. In that case, we can use `Selenium` to render the page and extract it.

Selium is a web automation tool that can simulate button clicking, form filling, page navigation, etc. A minimum to fill in Selium in a search bar is:

```python
from selenium import webdriver

chrome_options = Options()
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--no-sandbox")  # Avoid sandboxing (useful on Linux servers)
chrome_options.add_argument("--disable-dev-shm-usage")  # Prevent shared memory issues on Linux
driver = webdriver.Chrome(options=chrome_options)  # Ensure the Chrome driver is installed and in PATH

driver = webdriver.Chrome()  # Start the Chrome browser
driver.get("https://example.com")  # Open a webpage
WebDriverWait(driver, 100).until(lambda driver: driver.execute_script('return document.readyState') == 'complete')

element = driver.find_element("name", "q")  # Find a search bar by name
element.send_keys("Selenium")  # Type "Selenium" in the search bar
element.submit()  # Submit the form

driver.quit()  # Close the browser
```

- `headless` is important, otherwise we are able to see the chrome browser.
- `driver.get("https://example.com")` unblocks when page loading begins.
- `WebDriverWait(driver, 100).until(lambda driver: driver.execute_script('return document.readyState') == 'complete')` uses JS to check for the webpage's ready state

### HTML Header

To properly send an HTTP request, we need to specify a `User-Agent` header that identifies a web browser so that the script imitiates the behavior of a web browser. The reason is some webservers **may block clients that do not have such a header**.

```python
# Typical for Google Chrome
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
}
# Download and add the image
response = requests.get(car.image_url, headers=headers)
if response.status_code == 200:
    ...
else:
    print("Car image url fetching failed: ", car.image_url)
```
