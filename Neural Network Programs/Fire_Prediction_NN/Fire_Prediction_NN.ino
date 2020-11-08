/*
 * Written by Romesh Fernando (1940609) - romeshfernandodean98@gmail.com
 * UG(CS&SE) - UOB
 * ANT - AI based firefighting robot (NN for navigate robot)
 * Arduino v1.8.12
 */

/*
 * Include required libraries
 */

#include <DHT.h>
#include <DHT_U.h>

#include <math.h>
#include <Arduino.h>

#include "DHT.h"
#define DHTPIN 2  
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);


/*
 * Declaration of parameters (These parameters can be changed when configure the neural network for get better outputs)
 */

const int PatternCount = 27;
const int InputNodes = 3;
const int HiddenNodes = 4;
const int OutputNodes = 2;
const float LearningRate = 0.2;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.04;
int trainDone=0;


/*
 * Declaration of parameters (These parameters can be changed when configure the neural network for get better outputs)
 */

float Input[PatternCount][InputNodes] = {
  { 1, 0, 0 },
  { 0, 1, 0 }, 
  { 0, 0, 1 }, 
    
  { 1, 1, 0 },  
  { 0, 1, 1 }, 
  { 1, 0, 1 }, 
  
  { 1, 1, 1 },  
  { 0, 0, 0 }, 
  
  { 0.5, 0, 0 },
  { 0, 0.5, 0 }, 
  { 0, 0, 0.5 }, 
    
  { 0.5, 0.5, 0 },  
  { 0, 0.5, 0.5 }, 
  { 0.5, 0, 0.5 }, 
  
  { 0.5, 0.5, 0.5 },  
  { 0, 0, 0 }, 

  { 0.8, 0, 0 },
  { 0, 0.8, 0 }, 
  { 0, 0, 0.8 }, 
    
  { 0.8, 0.8, 0 },  
  { 0, 0.8, 0.8 }, 
  { 0.8, 0, 0.8 }, 
  
  { 0.8, 0.8, 0.8 },  
  { 0, 0, 0 },
  {1,0.7,1},
  
  {0.5,0.7,1},
  {0.5,0.9,0.8}

};

const float Target[PatternCount][OutputNodes] = {
  
   {0,1},
   {0,1},
   {0,1},
   
   {0,1},
   {0,1},
   {0,1},
    
   {1,0},
   {0,1},

   {0,1},
   {0,1},
   {0,1},
   
   {0,1},
   {0,1},
   {0,1},
    
   {0,1},
   {0,1},

   {0,1},
   {0,1},
   {0,1},
   
   {0,1},
   {0,1},
   {0,1},
    
   {1,0},
   {0,1},
   {1,0},
   
   {0,1},
   {0,1}
};


/*
 * Variable declaration
 */

int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

void setup(){
  dht.begin();
  Serial.begin(115200);
  ReportEvery1000 = 1;
  for( p = 0 ; p < PatternCount ; p++ ) {    
    RandomizedIndex[p] = p ;
  }

  
}  

// declare the input data array

float inData[1][4] = {
  { 0, 0, 0, 0},
  
};

float NNInput[4];

void loop (){

  if(trainDone==0)
  {
    train_NN();  
  }
  else
  {
    run_NN();  
  }
  
}


  /* 
   *  Run NN
   */
   
void run_NN()
{
    int in1 = analogRead(A1);          //flame light
    int in2 = analogRead(A2);          //smoke
    float in3 = dht.readTemperature(); //temperature
 
    in1 = map(in1, 350, 650, 0, 100);
    in2 = map(in2, 50, 150, 0, 100);
    in3 = map(in3, 33, 38, 0, 100);
    
    in1 = constrain(in1, 0, 100);
    in2 = constrain(in2, 0, 100);
    in3 = constrain(in3, 0, 100);

    NNInput[1] = float(in1) / 100;
    NNInput[2] = float(in2) / 100;
    NNInput[3] = float(in3) / 100;

    //remove comments of below three lines for testing with custom inputs (add 0, 1 or values between 0 and 1)
    //NNInput[1]=0;
    //NNInput[2]=0;
    //NNInput[3]=0;

    Serial.print ("Inputs ");
    Serial.print (NNInput[1], 1);
    Serial.print (" , ");
    Serial.print (NNInput[2], 1);
    Serial.print (" , ");
    Serial.print (NNInput[3], 1);
    Serial.println (" ");
    
    inData[1][0] = NNInput[1];
    inData[1][1] = NNInput[2];
    inData[1][2] = NNInput[3];


    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += inData[1][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    Serial.print ("Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
    
    if(Output[0]>=Output[1])
    {
      Serial.println (" Flame detected ");
      Serial.println (" ");
      analogWrite(9, 220);
      delay(2000);

    }else
    {
      Serial.println (" No flame ");
      Serial.println (" ");
      analogWrite(9, LOW);
      delay(1000);
    }
    
  }


  /* 
   *  Train NN
   */

void train_NN(){

  for( i = 0 ; i < HiddenNodes ; i++ ) {    
    for( j = 0 ; j <= InputNodes ; j++ ) { 
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100))/100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  for( i = 0 ; i < OutputNodes ; i ++ ) {    
    for( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;  
      Rando = float(random(100))/100;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {    
    for( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p] ; 
      RandomizedIndex[p] = RandomizedIndex[q] ; 
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;

    for( q = 0 ; q < PatternCount ; q++ ) {    
      p = RandomizedIndex[q];

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = HiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
      }

      for( i = 0 ; i < OutputNodes ; i++ ) {    
        Accum = OutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }

      for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

      for( i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

    Serial.print ("  Error = ");
    Serial.println (Error, 5);
   

    if( Error < Success )
    {
      trainDone=1;
    }
    if( Error < Success )break;
  }
  

}


  
