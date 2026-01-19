/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdio.h"
#include "string.h"
#include "stm32u5xx_ll_gpio.h"

// ADC Buffers
uint16_t Val[128] = {0};
uint16_t Val_adc14[4] = {0};

volatile uint32_t convCompleted_1 = 0;
volatile float hf_max_result = 0.0f;
volatile float mf_max_result = 0.0f;
volatile float lf_max_result = 0.0f;

// Filter coefficients
//float hf_a1 = 1.48799279f, hf_a2 = 0.69728902f, hf_b0 = 0.15135549f, hf_b2 = -0.15135549f;
//float hf_a1 = 1.78909292f, hf_a2 = 0.93178209f, hf_b0 = 0.03410896f, hf_b2 = -0.03410896f;
float hf_a1 = 1.92868856, hf_a2 = 0.96530986, hf_b0 = 0.01734507, hf_b2 = -0.01734507;

float lf_a1 = 0.72111508f, lf_a2 = 0.41071397f, lf_b0 = 0.29464302f, lf_b2 = -0.29464302f;
//float mf_a1 = -1.38236825f, mf_a2 = 0.83743011f, mf_b0 = 0.08128495f, mf_b2 = -0.08128495f;
//float mf_a1 = -1.60599656f, mf_a2 = 0.98250454, mf_b0 = 0.00874773f, mf_b2 = -0.00874773f;
//float mf_a1 = -1.64602244f, mf_a2 = 0.98250454, mf_b0 = 0.00874773f, mf_b2 = -0.00874773f;
float mf_a1 = -1.690599656, mf_a2 = 0.98250454, mf_b0 = 0.00874773f, mf_b2 = -0.00874773f;

volatile float hf_result = 0.0f, mf_result = 0.0f, lf_result = 0.0f;
float hf_z1 = 0.0f, hf_z2 = 0.0f;
float lf_z1 = 0.0f, lf_z2 = 0.0f;
float mf_z1 = 0.0f, mf_z2 = 0.0f;

// Large transmission buffer
#define TX_BUFFER_SIZE 2048
int16_t tx_buffer[TX_BUFFER_SIZE/2]; // int16_t buffer
volatile uint16_t tx_buffer_index = 0;
volatile uint32_t total_samples_copied = 0;

// ANN Configuration
#define INPUT_CHANNELS 4
#define INTERRUPT_FREQUENCY 15000
#define PROCESS_INTERVAL_MS 10
#define SAMPLES_PER_PROCESS ((INTERRUPT_FREQUENCY * PROCESS_INTERVAL_MS) / 1000)
#define COMPRESSION_FACTOR 3
#define COMPRESSED_SAMPLES (SAMPLES_PER_PROCESS / COMPRESSION_FACTOR)

// Double buffering for ANN to prevent race conditions
#define ANN_BUFFER_COUNT 2
volatile float compressed_features[ANN_BUFFER_COUNT][INPUT_CHANNELS][COMPRESSED_SAMPLES];
volatile uint8_t ann_buffer_ready[ANN_BUFFER_COUNT] = {0, 0};
volatile uint8_t ann_write_buffer = 0;
volatile uint8_t ann_read_buffer = 0;

// Circular buffer for raw data (2 cycles for difference calculation)
#define HISTORY_SIZE SAMPLES_PER_PROCESS
volatile float data_history[INPUT_CHANNELS][HISTORY_SIZE * 2];
volatile uint16_t history_index = 0;

// Compressed buffer indices
volatile uint8_t compress_counter = 0;
volatile uint16_t compress_index = 0;

// Neural network structure
#define LAYER1_SIZE 25
#define LAYER2_SIZE 25
#define LAYER3_SIZE 20
#define OUTPUT_SIZE 1

// Neural network weights and biases - placeholder
float w1[LAYER1_SIZE][INPUT_CHANNELS * COMPRESSED_SAMPLES] = {0.4f};
float b1[LAYER1_SIZE] = {0.4f};
float w2[LAYER2_SIZE][LAYER1_SIZE] = {0.4f};
float b2[LAYER2_SIZE] = {0.4f};
float w3[LAYER3_SIZE][LAYER2_SIZE] = {0.4f};
float b3[LAYER3_SIZE] = {0.4f};
float w4[OUTPUT_SIZE][LAYER3_SIZE] = {0.4f};
float b4[OUTPUT_SIZE] = {0.4f};
float ann_output = 0;

// Integrator
#define window_size 300
uint32_t moving_window[window_size] = {0};
uint32_t average_acc = 0;
uint16_t current_index = 0;
int32_t integ_acc = 0;
uint32_t integ_offset = 0;
uint8_t integ_flag_full = 0; // Initialize flag to 0
// Define the known startup DC offset
#define STARTUP_OFFSET 7500<<4
#define scale_factor 4

// Neural network processing function
void process_ann(void) {
    // Check if there's data to process
    if (ann_buffer_ready[ann_read_buffer]) {
        float layer1_out[LAYER1_SIZE];
        float layer2_out[LAYER2_SIZE];
        float layer3_out[LAYER3_SIZE];
        float output;

        // First layer
        for (int i = 0; i < LAYER1_SIZE; i++) {
            float sum = b1[i];
            int weight_idx = 0;

            // Add weighted inputs from compressed features
            for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
                for (int j = 0; j < COMPRESSED_SAMPLES; j++) {
                    sum += w1[i][weight_idx++] * compressed_features[ann_read_buffer][ch][j];
                }
            }

            layer1_out[i] = (sum > 0) ? sum : 0;
        }

        // Second layer
        for (int i = 0; i < LAYER2_SIZE; i++) {
            float sum = b2[i];
            for (int j = 0; j < LAYER1_SIZE; j++) {
                sum += w2[i][j] * layer1_out[j];
            }
            layer2_out[i] = (sum > 0) ? sum : 0;
        }

        // Third layer
        for (int i = 0; i < LAYER3_SIZE; i++) {
            float sum = b3[i];
            for (int j = 0; j < LAYER2_SIZE; j++) {
                sum += w3[i][j] * layer2_out[j];
            }
            layer3_out[i] = (sum > 0) ? sum : 0;
        }

        // Output layer
        output = b4[0];
        for (int i = 0; i < LAYER3_SIZE; i++) {
            output += w4[0][i] * layer3_out[i];
        }
        ann_output = 1.0f / (1.0f + expf(-output));

        // Mark buffer as processed
        ann_buffer_ready[ann_read_buffer] = 0;
        ann_read_buffer = (ann_read_buffer + 1) % ANN_BUFFER_COUNT;
    }
}


/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
void MHF_Filter(uint16_t *data )
{

	uint16_t ind = 0 ;

		 //mf_z1 = 0.0f;
		// mf_z2 = 0.0f ;


	for(ind=0 ; ind< 64 ; ind+= 2 )

	{		//filter hf
		hf_result = hf_b0 * data[ind] + hf_z1;
		//if ( hf_result > hf_max_result ) hf_max_result =  hf_result ;
		//if(hf_result >0) hf_max_result += hf_result ;
	//	if(ind%8 == 0 ) hf_max_result += fabs(hf_result)*0.05f ; // high level hf
	//	if(ind%16 == 0 ) {
			if(hf_result >0) hf_max_result += (hf_result) ;
			else hf_max_result -= (hf_result) ;
	//	}
		hf_z1 =  hf_z2 - hf_a1 * hf_result;
		hf_z2 = hf_b2 * data[ind] - hf_a2 * hf_result;
		hf_result = hf_b0 * data[ind+1] + hf_z1;

	//	if(ind%16 == 0 ) {
			if(hf_result >0) hf_max_result += (hf_result) ;
			else hf_max_result -= (hf_result) ;
	//	}
		//if ( hf_result > hf_max_result ) hf_max_result =  hf_result ;
		//hf_max_result +=  hf_result ;
		hf_z1 =  hf_z2 - hf_a1 * hf_result;
		hf_z2 = hf_b2 * data[ind+1] - hf_a2 * hf_result;

			//filter mf
		mf_result = mf_b0 * data[ind]+ mf_z1;
		//if ( mf_result > mf_max_result ) mf_max_result =  mf_result ;
		//if ( mf_result < mf_min_result ) mf_min_result =  mf_result ;
		//if( (mf_result >0)&&(ind>32) ) mf_max_result += mf_result ;
		//if(ind%8 == 0 ) mf_max_result += fabs(mf_result)*0.03f ;
	//	if(ind%16 == 0 ) {
			if(mf_result >0) mf_max_result += (mf_result) ;
			else mf_max_result -= (mf_result) ;
	//	}

		mf_z1 =  mf_z2 - mf_a1 * mf_result;
		mf_z2 = mf_b2 * data[ind] - mf_a2 * mf_result;

		mf_result = mf_b0 * data[ind+1]+ mf_z1;
	//	if(ind%16 == 0 ) {
			if(mf_result >0) mf_max_result += (mf_result) ;
			else mf_max_result -= (mf_result) ;
	//	}

		//if ( mf_result > mf_max_result ) mf_max_result =  mf_result ;
		//if ( mf_result < mf_min_result ) mf_min_result =  mf_result ;
		//mf_max_result += mf_result ;
		mf_z1 =  mf_z2 - mf_a1 * mf_result;
		mf_z2 = mf_b2 * data[ind+1] - mf_a2 * mf_result;

		// filter lf
/*
		lf_result = lf_b0 * data[ind+1] + lf_z1;
		lf_max_result += lf_result ;
		lf_z1 =  lf_z2 - lf_a1 * lf_result;
		lf_z2 = lf_b2 * data[ind+1] - lf_a2 * lf_result; */
	}

	//hf_max_result = hf_max_result +  hf_result*0.001f ;
	//convCompleted_1 = convCompleted_1 -1 ;




}
// Simplified feature extraction in interrupt
void extract_features_simple(float ch0, float ch1, float ch2, float ch3) {
    static float sum[INPUT_CHANNELS] = {0};

    // Accumulate for compression
    sum[0] += fabsf(ch0);
    sum[1] += fabsf(ch1);
    sum[2] += fabsf(ch2);
    sum[3] += fabsf(ch3);

    compress_counter++;

    // Compress when enough samples accumulated
    if (compress_counter >= COMPRESSION_FACTOR) {
        // Store compressed data
        if (compress_index < COMPRESSED_SAMPLES) {
            compressed_features[ann_write_buffer][0][compress_index] = sum[0];
            compressed_features[ann_write_buffer][1][compress_index] = sum[1];
            compressed_features[ann_write_buffer][2][compress_index] = sum[2];
            compressed_features[ann_write_buffer][3][compress_index] = sum[3];
        }

        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;
        sum[3] = 0;
        compress_counter = 0;
        compress_index++;

        // When buffer is full, mark it as ready and switch to next buffer
        if (compress_index >= COMPRESSED_SAMPLES) {
            ann_buffer_ready[ann_write_buffer] = 1;
            ann_write_buffer = (ann_write_buffer + 1) % ANN_BUFFER_COUNT;
            compress_index = 0;
           // mf_max_result = 0.0f;
           // lf_max_result = 0.0f;
            //hf_max_result = 0.0f;
        }
    }
}


void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance == ADC4) {
        MHF_Filter(&Val[64]);
    }

    if (hadc->Instance == ADC1) {
        convCompleted_1++;

        // LF filter
        lf_result = lf_b0 * Val_adc14[2] + lf_z1;
        lf_max_result += (lf_result > 0) ? lf_result : -lf_result;
        lf_z1 = lf_z2 - lf_a1 * lf_result;
        lf_z2 = lf_b2 * Val_adc14[2] - lf_a2 * lf_result;

        // Store in history buffer
        data_history[0][history_index] = (float)Val_adc14[0];
        data_history[1][history_index] = lf_result;
        data_history[2][history_index] = mf_max_result;
        data_history[3][history_index] = hf_max_result;

        // Get previous cycle data (HISTORY_SIZE samples ago)
        uint16_t prev_index = (history_index + HISTORY_SIZE) % (HISTORY_SIZE * 2);
        float diff0 = data_history[0][history_index] - data_history[0][prev_index];
        float diff1 = data_history[1][history_index] - data_history[1][prev_index];
        float diff2 = data_history[2][history_index] - data_history[2][prev_index];
        float diff3 = data_history[3][history_index] - data_history[3][prev_index];

        // Extract features with differences
        extract_features_simple(diff0, diff1, diff2, diff3);

        // Copy data directly to transmission buffer in interrupt (simple int16_t copy)
        if ((tx_buffer_index + 4) <= (TX_BUFFER_SIZE/2)) {
            tx_buffer[tx_buffer_index] =   (int16_t) ann_buffer_ready[ann_read_buffer] ; // Val_adc14[1]   ;// (integ_acc >> scale_factor);// (integ_acc >> scale_factor) ; // ann_buffer_ready[ann_read_buffer] ;  // (integ_acc >> scale_factor) ; //ann_buffer_ready[ann_read_buffer] ; // history_index; // ann_buffer_ready[ann_read_buffer] ;// (ann_output*100) ; //  Val_adc14[2] ; //(ann_output);
            tx_buffer[tx_buffer_index + 1] = (int16_t)lf_max_result;
            tx_buffer[tx_buffer_index + 2] = (int16_t)mf_max_result; ///Val_adc14[0];
            tx_buffer[tx_buffer_index + 3] = (int16_t)hf_max_result;
            tx_buffer_index += 4;
            total_samples_copied++;
        }
        lf_max_result = 0 ;
        mf_max_result = 0 ;
        hf_max_result = 0 ;

        // Update history index
        history_index = (history_index + 1) % HISTORY_SIZE;
    }
}



void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance == ADC4) {
        MHF_Filter(&Val[0]);
    }

    if (hadc->Instance == ADC1) {
        // Filter
        lf_result = lf_b0 * Val_adc14[0] + lf_z1;
        lf_max_result += (lf_result > 0) ? lf_result : -lf_result;
        lf_z1 = lf_z2 - lf_a1 * lf_result;
        lf_z2 = lf_b2 * Val_adc14[0] - lf_a2 * lf_result;
        // Integrator
        uint32_t new_val_scaled = (uint32_t)Val_adc14[0] << scale_factor;
        // Add new value to the sum
        average_acc +=  new_val_scaled ;//Val_adc14[0];
        // Remove old value from the sum
        average_acc -= moving_window[current_index];
        // Update the window with the new value
        moving_window[current_index] =  new_val_scaled ; //Val_adc14[0];
        // Update the circular buffer index
        current_index++;
        if (current_index == window_size) {
            current_index = 0;
            // The window is now full for the first time, set the flag
            integ_flag_full = 1;
        }

        // Use a conditional check to choose the correct offset for the integrator
        if (integ_flag_full == 0) {
            // Startup period: use the fixed, known DC offset
            integ_acc = new_val_scaled + integ_acc - STARTUP_OFFSET;
        } else {
            // Steady state: use the calculated moving average as the DC offset
            integ_offset = average_acc / window_size;
            integ_acc = new_val_scaled + integ_acc - integ_offset;
            integ_acc -= (integ_acc >> 10);
        }
    }
}

/*
#define window_size 150
uint16_t moving_window[window_size] = {0} ;
uint32_t average_acc = 0 ;
uint16_t current_index = 0 ;
int32_t integ_acc = 0 ;
uint16_t integ_offset = 0 ;
uint8_t integ_flag_full = 0 ;

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance == ADC4) {
        MHF_Filter(&Val[0]);
    }

    if (hadc->Instance == ADC1) {
        lf_result = lf_b0 * Val_adc14[0] + lf_z1;
        lf_max_result += fabsf(lf_result) * 0.08f;
        lf_z1 = lf_z2 - lf_a1 * lf_result;
        lf_z2 = lf_b2 * Val_adc14[0] - lf_a2 * lf_result;

        // Integrator
        average_acc += Val_adc14[0] ;
        //Remove old value, since we init with 0 we can remove even before the window full
        average_acc -= moving_window[current_index] ;
        moving_window[current_index] = Val_adc14[0] ;
        current_index++;
        if ( current_index ==window_size ) current_index = 0 ;
        //Compute offset
        integ_offset = average_acc / window_size ;
        integ_acc = Val_adc14[0] +  integ_acc - integ_offset ;
    }
} */



/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
ADC_HandleTypeDef hadc4;
DMA_NodeTypeDef Node_GPDMA1_Channel0;
DMA_QListTypeDef List_GPDMA1_Channel0;
DMA_HandleTypeDef handle_GPDMA1_Channel0;
DMA_NodeTypeDef Node_GPDMA1_Channel1;
DMA_QListTypeDef List_GPDMA1_Channel1;
DMA_HandleTypeDef handle_GPDMA1_Channel1;

DCACHE_HandleTypeDef hdcache1;

TIM_HandleTypeDef htim1;

UART_HandleTypeDef huart4;
DMA_HandleTypeDef handle_GPDMA1_Channel2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void SystemPower_Config(void);
static void MX_GPIO_Init(void);
static void MX_GPDMA1_Init(void);
static void MX_ADC1_Init(void);
static void MX_ADC4_Init(void);
static void MX_TIM1_Init(void);
static void MX_ICACHE_Init(void);
static void MX_DCACHE1_Init(void);
static void MX_UART4_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the System Power */
  SystemPower_Config();

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_GPDMA1_Init();
  MX_ADC1_Init();
  MX_ADC4_Init();
  MX_TIM1_Init();
  MX_ICACHE_Init();
  MX_DCACHE1_Init();
  MX_UART4_Init();
  /* USER CODE BEGIN 2 */
  memset(moving_window, 0, sizeof(moving_window));
  HAL_TIM_Base_Start(&htim1) ;
    // HAL_ADC_Start_IT(&hadc3);
  HAL_ADC_Start_DMA(&hadc4, ( uint16_t*) Val , sizeof (Val) / sizeof(Val[0]));
  HAL_ADC_Start_DMA(&hadc1, ( uint16_t*) Val_adc14 , sizeof (Val_adc14) / sizeof(Val_adc14[0]));

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  //ABCDEFGHIJ in ASCII code

  uint32_t last_conv_count = 0;
  uint32_t conv_count_threshold = 20; // Send every 20 conversions
  while (1) {
      // Process ANN when data is available
      process_ann();

      // Send data based on conversion count threshold
      if ((convCompleted_1 - last_conv_count) >= conv_count_threshold) {
          last_conv_count = convCompleted_1;

          // Send data if buffer has content
          if (tx_buffer_index > 0) {
              HAL_UART_Transmit_DMA(&huart1, (uint8_t*)tx_buffer, tx_buffer_index * 2);
              tx_buffer_index = 0;
          }

          // Periodic reset

      }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_0;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLMBOOST = RCC_PLLMBOOST_DIV4;
  RCC_OscInitStruct.PLL.PLLM = 3;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 1;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLLVCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the common periph clock
  */
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADCDAC;
  PeriphClkInit.AdcDacClockSelection = RCC_ADCDACCLKSOURCE_PLL2;
  PeriphClkInit.PLL2.PLL2Source = RCC_PLLSOURCE_HSI;
  PeriphClkInit.PLL2.PLL2M = 1;
  PeriphClkInit.PLL2.PLL2N = 21;
  PeriphClkInit.PLL2.PLL2P = 2;
  PeriphClkInit.PLL2.PLL2Q = 2;
  PeriphClkInit.PLL2.PLL2R = 8;
  PeriphClkInit.PLL2.PLL2RGE = RCC_PLLVCIRANGE_1;
  PeriphClkInit.PLL2.PLL2FRACN = 0;
  PeriphClkInit.PLL2.PLL2ClockOut = RCC_PLL2_DIVR;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Power Configuration
  * @retval None
  */
static void SystemPower_Config(void)
{

  /*
   * Switch to SMPS regulator instead of LDO
   */
  if (HAL_PWREx_ConfigSupply(PWR_SMPS_SUPPLY) != HAL_OK)
  {
    Error_Handler();
  }
/* USER CODE BEGIN PWR */
/* USER CODE END PWR */
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_14B;
  hadc1.Init.GainCompensation = 0;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 2;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_EXTERNALTRIG_T1_TRGO;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.TriggerFrequencyMode = ADC_TRIGGER_FREQ_HIGH;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DMA_CIRCULAR;
  hadc1.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_12;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_6CYCLES;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_10;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief ADC4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC4_Init(void)
{

  /* USER CODE BEGIN ADC4_Init 0 */

  /* USER CODE END ADC4_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC4_Init 1 */

  /* USER CODE END ADC4_Init 1 */

  /** Common config
  */
  hadc4.Instance = ADC4;
  hadc4.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc4.Init.Resolution = ADC_RESOLUTION_12B;
  hadc4.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc4.Init.ScanConvMode = ADC4_SCAN_DISABLE;
  hadc4.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc4.Init.LowPowerAutoPowerOff = ADC_LOW_POWER_NONE;
  hadc4.Init.LowPowerAutoWait = DISABLE;
  hadc4.Init.ContinuousConvMode = ENABLE;
  hadc4.Init.NbrOfConversion = 1;
  hadc4.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc4.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc4.Init.DMAContinuousRequests = ENABLE;
  hadc4.Init.TriggerFrequencyMode = ADC_TRIGGER_FREQ_LOW;
  hadc4.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc4.Init.SamplingTimeCommon1 = ADC4_SAMPLETIME_12CYCLES_5;
  hadc4.Init.SamplingTimeCommon2 = ADC4_SAMPLETIME_1CYCLE_5;
  hadc4.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init(&hadc4) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_11;
  sConfig.Rank = ADC4_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC4_SAMPLINGTIME_COMMON_1;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC4_Init 2 */

  /* USER CODE END ADC4_Init 2 */

}

/**
  * @brief DCACHE1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_DCACHE1_Init(void)
{

  /* USER CODE BEGIN DCACHE1_Init 0 */

  /* USER CODE END DCACHE1_Init 0 */

  /* USER CODE BEGIN DCACHE1_Init 1 */

  /* USER CODE END DCACHE1_Init 1 */
  hdcache1.Instance = DCACHE1;
  hdcache1.Init.ReadBurstType = DCACHE_READ_BURST_WRAP;
  if (HAL_DCACHE_Init(&hdcache1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DCACHE1_Init 2 */

  /* USER CODE END DCACHE1_Init 2 */

}

/**
  * @brief GPDMA1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPDMA1_Init(void)
{

  /* USER CODE BEGIN GPDMA1_Init 0 */

  /* USER CODE END GPDMA1_Init 0 */

  /* Peripheral clock enable */
  __HAL_RCC_GPDMA1_CLK_ENABLE();

  /* GPDMA1 interrupt Init */
    HAL_NVIC_SetPriority(GPDMA1_Channel0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(GPDMA1_Channel0_IRQn);
    HAL_NVIC_SetPriority(GPDMA1_Channel1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(GPDMA1_Channel1_IRQn);
    HAL_NVIC_SetPriority(GPDMA1_Channel2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(GPDMA1_Channel2_IRQn);

  /* USER CODE BEGIN GPDMA1_Init 1 */

  /* USER CODE END GPDMA1_Init 1 */
  /* USER CODE BEGIN GPDMA1_Init 2 */

  /* USER CODE END GPDMA1_Init 2 */

}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache in 1-way (direct mapped cache)
  */
  if (HAL_ICACHE_ConfigAssociativityMode(ICACHE_1WAY) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 5333;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief UART4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_UART4_Init(void)
{

  /* USER CODE BEGIN UART4_Init 0 */

  /* USER CODE END UART4_Init 0 */

  /* USER CODE BEGIN UART4_Init 1 */

  /* USER CODE END UART4_Init 1 */
  huart4.Instance = UART4;
  huart4.Init.BaudRate = 2000000;
  huart4.Init.WordLength = UART_WORDLENGTH_8B;
  huart4.Init.StopBits = UART_STOPBITS_1;
  huart4.Init.Parity = UART_PARITY_NONE;
  huart4.Init.Mode = UART_MODE_TX_RX;
  huart4.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart4.Init.OverSampling = UART_OVERSAMPLING_16;
  huart4.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart4.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart4.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart4, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart4, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN UART4_Init 2 */

  /* USER CODE END UART4_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */
  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_BUTTON_Pin */
  GPIO_InitStruct.Pin = USER_BUTTON_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_BUTTON_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LED_GREEN_Pin */
  GPIO_InitStruct.Pin = LED_GREEN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED_GREEN_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
