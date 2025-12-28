import pickle
from fastapi import FastAPI
from routers import predict
from contextlib import asynccontextmanager

# Lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the ML model
    print("Loading ML model")
    with open("models/ensemble_model.pkl", "rb") as f:
        app.state.model = pickle.load(f)
    print("ML model loaded!")

    yield

    # Cleanup
    app.state.model = None
    print("API shutdown complete")


app = FastAPI(
    title="Loan Defaulter API",
    description="Classify an applicant as Defaulter/Non-defaulter based on financial details",
    version="0.1.0",
    root_path="/api/v1",
    lifespan=lifespan
)

# Include the routers
app.include_router(predict.router)