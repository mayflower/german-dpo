# german-dpo
A set of german pdo dataset translations and tools to create them

# Prepare configuration
```
cp .env.dev .env
```
Add, if necessary, the API_KEY and change the models according to your requirements

# Create new environment
```
make build
```

# Start app
```
make up
```

# Generate product descriptions based on product images
```
make generate_on_image
```
