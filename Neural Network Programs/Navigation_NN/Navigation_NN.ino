/*
 * Written by Romesh Fernando (1940609) - romeshfernandodean98@gmail.com
 * UG(CS&SE) - UOB
 * ANT - AI based firefighting robot (NN for navigate robot)
 * Arduino v1.8.12
 */

#include <math.h>

/*
 * Declaration of parameters (These parameters can be changed when configure the neural network for get better outputs)
 */
 
const int dataSetCount = 16;
const int inputNodes = 4;
const int hiddenNodes = 5;
const int outputNodes = 4;
const float learningRate = 0.3;
const float Momentum = 0.9;
const float weight = 0.5;
const float successRate = 0.03;
int trainDone=0;
int lft;
int rgt;


/*
 * Taking sensor inputs
 */

long dis1(int triggerPin, int echoPin)
{
  pinMode(triggerPin, OUTPUT); 
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(50);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  return pulseIn(echoPin, HIGH)*0.01723;
}

long dis2(int triggerPin, int echoPin)
{
  pinMode(triggerPin, OUTPUT); 
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(50);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  return pulseIn(echoPin, HIGH)*0.01723;
}

long dis3(int triggerPin, int echoPin)
{
  pinMode(triggerPin, OUTPUT); 
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(50);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  return pulseIn(echoPin, HIGH)*0.01723;
}

long dis4(int triggerPin, int echoPin)
{
  pinMode(triggerPin, OUTPUT);  
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(50);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  return pulseIn(echoPin, HIGH)*0.01723;
}


/*
 * Training data
 */

float Input[dataSetCount][inputNodes] = {
  
  { 0, 1, 1, 0 },
  { 0, 1, 0, 0 }, 
  { 1, 1, 1, 0 }, 
    
  { 1, 1, 0, 0 },  
  { 0, 0, 1, 0 }, 
  { 1, 0, 0, 0 }, 
  
  { 0, 0, 0, 0 },  
  { 0, 0, 0, 1 }, 
  { 0, 1, 0, 1 }, 
    
  { 0, 0, 1, 1 }, 
  { 0, 1, 1, 1 }, 
  { 1, 0, 0, 1 }, 
    
  { 1, 1, 0, 1 },
  { 1, 0, 1, 1 }, 
  { 1, 0, 1, 0 },  
  { 1, 1, 1, 1 },  

};

const float Target[dataSetCount] [outputNodes] = {
  
  { 0,1,0,0 },   
  { 0,1,0,0 },    
  { 1,0,0,0 },     
    
  { 1,0,0,0 },      
  { 0,0,1,0 },    
  { 1,0,0,0},     
    
  { 0,0,0,0 },    
  { 0,0,0,1 },
  { 0,1,0,0 },
    
  { 0,0,1,0 },
  { 0,1,0,0 },
  {1,0,0,0 },
    
  { 1,0,0,0 }, 
  { 1,0,0,0 },
  { 1,0,0,0 },
  { 1,0,0,0},

};


/*
 * Variable declaration
 */

int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[dataSetCount];
long  trainingCycleNo;
float Rando;
float Error;
float Accum;

float Hidden[hiddenNodes];
float Output [outputNodes];
float HiddenWeights[inputNodes+1][hiddenNodes]; // inputNodes+1 = input nodes + bias node
float OutputWeights[hiddenNodes+1] [outputNodes]; // hiddenNodes+1 = input nodes + bias node
float HiddenDelta[hiddenNodes];
float OutputDelta [outputNodes];
float ChangeHiddenWeights[inputNodes+1][hiddenNodes];
float ChangeOutputWeights[hiddenNodes+1] [outputNodes];

void setup(){
  Serial.begin(115200);

  //take random entry from training dataset
  
  for( p = 0 ; p < dataSetCount ; p++ ) {    
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


  long DIS3 = dis1(A3,A4);   // Collect distances.
  long DIS2 = dis2(A1,A2);   // Collect distances.
  long DIS1 = dis3(A0,A5);   // Collect distances.
  long DIS4 = dis4(10,11);   // Collect distances.


  /* 
   *  Convert inputs into values between 0 and 100
   */
   
  DIS1 = map(DIS1, 30, 40, 0, 100);   
  DIS2 = map(DIS2, 30, 40, 0, 100);
  DIS3 = map(DIS3, 30, 40, 0, 100);
  DIS4 = map(DIS4, 30, 40, 0, 100);


  /* 
   *  Get a constrain of DIS1, DIS2, DIS3 and DIS4 
   */
  
  DIS1 = constrain(DIS1, 0, 100);
  DIS2 = constrain(DIS2, 0, 100);
  DIS3 = constrain(DIS3, 0, 100);
  DIS4 = constrain(DIS4, 0, 100);


  /* 
   *  Get a final input values 
   */
   
  NNInput[0] = float(DIS1) / 100;
  NNInput[1] = float(DIS2) / 100;
  NNInput[2] = float(DIS3) / 100;
  NNInput[3] = float(DIS4) / 100;

  //remove comments of below three lines for testing with custom inputs (add 0, 1 or values between 0 and 1)
  //NNInput[0]=1;
  //NNInput[1]=0;
  //NNInput[2]=0;
  //NNInput[3]=0;

  Serial.print ("Inputs ");
  Serial.print (NNInput[0], 1);
  Serial.print (" , ");
  Serial.print (NNInput[1], 1);
  Serial.print (" , ");
  Serial.print (NNInput[2], 1);
  Serial.print (" , ");
  Serial.print (NNInput[3], 1);
  Serial.println (" ");
  
  inData[1][0] = NNInput[0];
  inData[1][1] = NNInput[1];
  inData[1][2] = NNInput[2];
  inData[1][3] = NNInput[3];

  /* 
   *  Calculate activations of hidden nodes and get final outputs
   */

  for( i = 0 ; i < hiddenNodes ; i++ ) {    
    Accum = HiddenWeights[inputNodes][i] ; // from bias node
    for( j = 0 ; j < inputNodes ; j++ ) {
      Accum += inData[1][j] * HiddenWeights[j][i] ;
    }
    Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;  // Activation function
  }


  /* 
   *  Calculate activations of output nodes
   */

  for( i = 0 ; i < outputNodes ; i++ ) {    
    Accum = OutputWeights[hiddenNodes][i] ; //from bias node
    for( j = 0 ; j < hiddenNodes ; j++ ) {
      Accum += Hidden[j] * OutputWeights[j][i] ;
    }
    Output[i] = 1.0/(1.0 + exp(-Accum)) ;  // Activation function
  }


   /* 
   *  Send signals to motors
   */
  
    int flame=digitalRead(13);  //Check whether the flame sensor has detected a flame
    if(flame==0)
    {
      Serial.println ("Flame sensor detected a flame");
      digitalWrite(7, LOW);
      digitalWrite(6, LOW);
      digitalWrite(9, LOW);
      digitalWrite(8, LOW);
    }
    else
    {
      if(Output[0]>=Output[1] && Output[0]>=Output[2] && Output[0]>=Output[3])
      {
        Serial.println ("Move forward");
        delay(1000);
        motorB(1); 
        motorA(1);   
      }
      if(Output[1]>=Output[0] && Output[1]>=Output[2] && Output[1]>=Output[3])
      {
        Serial.println ("Move left");
        delay(1000);
        motorB(1); 
        motorA(0);       
      }
      if(Output[2]>=Output[1] && Output[2]>=Output[0] && Output[2]>=Output[3])
      {
        Serial.println ("Move right");
        delay(1000);
        motorB(0); 
        motorA(1);       
      }
      if(Output[03]>=Output[1] && Output[3]>=Output[2] && Output[3]>=Output[0])
      {
        Serial.println ("Move backward");
        delay(1000);-
        motorB(0);
        motorA(0);       
      }
    }
}


  /* 
   *  Train NN
   */

void train_NN(){

  for( i = 0 ; i < hiddenNodes ; i++ ) {    
    for( j = 0 ; j <= inputNodes ; j++ ) { 
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100))/100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * weight ;
    }
  }

  for( i = 0 ; i < outputNodes ; i ++ ) {    
    for( j = 0 ; j <= hiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;  
      Rando = float(random(100))/100;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * weight ;
    }
  }

  for( trainingCycleNo = 1 ; trainingCycleNo < 2147483647 ; trainingCycleNo++) {    
    for( p = 0 ; p < dataSetCount ; p++) {
      q = random(dataSetCount);
      r = RandomizedIndex[p] ; 
      RandomizedIndex[p] = RandomizedIndex[q] ; 
      RandomizedIndex[q] = r ;
    }
    Error = 0.0 ;
    
    for( q = 0 ; q < dataSetCount ; q++ ) {    
      p = RandomizedIndex[q];

      for( i = 0 ; i < hiddenNodes ; i++ ) {    
        Accum = HiddenWeights[inputNodes][i] ;
        for( j = 0 ; j < inputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
      }

      for( i = 0 ; i < outputNodes ; i++ ) {    
        Accum = OutputWeights[hiddenNodes][i] ;
        for( j = 0 ; j < hiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

      for( i = 0 ; i < hiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < outputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }

      for( i = 0 ; i < hiddenNodes ; i++ ) {     
        ChangeHiddenWeights[inputNodes][i] = learningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[inputNodes][i] ;
        HiddenWeights[inputNodes][i] += ChangeHiddenWeights[inputNodes][i] ;
        for( j = 0 ; j < inputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = learningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

      for( i = 0 ; i < outputNodes ; i ++ ) {    
        ChangeOutputWeights[hiddenNodes][i] = learningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[hiddenNodes][i] ;
        OutputWeights[hiddenNodes][i] += ChangeOutputWeights[hiddenNodes][i] ;
        for( j = 0 ; j < hiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = learningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

    Serial.print (" Error ");
    Serial.println (Error, 5);
    
  if( Error < successRate )
    {
      trainDone=1;
    }
    if( Error < successRate )break;
  }
  
  }


  /* 
   * Control motor A
   */

void motorA(int dir1) {
  if (dir1 == 1) {

    //Serial.println("Driving Forward A: ");
    
    digitalWrite(9, HIGH);
    digitalWrite(8, LOW);
    if(rgt>lft)
    {
      analogWrite(3,170);
    }
    else
    {
      analogWrite(3,140);
    }
  }
  if (dir1 == 0) {
    //Serial.println("Driving Back A: ");
    digitalWrite(9, LOW);
    digitalWrite(8, HIGH);
    analogWrite(3,150);

   delay(700);
  }

  if (dir1 == -1) {
    Serial.println("Stop A: ");
    digitalWrite(9, LOW);
    digitalWrite(8, LOW);
    analogWrite(3,0);

    delay(500);
  }
}


  /* 
   * Control motor B
   */
   
void motorB(int dir) {
   if(dir == 1) {

    //Serial.println("Driving Forward B: ");
    
    digitalWrite(7, HIGH);
    digitalWrite(6, LOW);
    
    if(rgt>lft)
    {
      analogWrite(5,200);
    }
    else
    {
      analogWrite(5,160);
    }
  
  }
  if (dir == 0) {
    //Serial.println("Driving Back B: ");
    digitalWrite(7, LOW);
    digitalWrite(6, HIGH);
    analogWrite(5,150);

    delay(900);
  }
  if (dir == -1) {
    Serial.println("Stop B: ");
    digitalWrite(7, LOW);
    digitalWrite(6, LOW);
    analogWrite(5,0);

    delay(500);
  }
  
}
