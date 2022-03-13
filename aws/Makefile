# Change the values depends on situation
PREFIX:=souhub-fastapi-tutorial

PROFILE:=default
REGION:=ap-northeast-1
# ENV:=staging
ENV:=production
AWS_COMMAND_PREFIX := docker compose run --rm api

# Amazon Linux2 AMI
AMI_IMAGE_ID := ami-0404778e217f54308

vpc:
	$(AWS_COMMAND_PREFIX) cloudformation validate-template \
		--profile ${PROFILE} \
		--template-body file://templates/vpc.yml && \
	$(AWS_COMMAND_PREFIX) cloudformation deploy \
		--profile ${PROFILE} \
		--template-file ./templates/vpc.yml \
		--stack-name $(PREFIX)-$(ENV)-vpc \
		--region $(REGION) \
		--parameter-overrides \
		Prefix=$(PREFIX) \
		Environment=$(ENV) \
		Region=${REGION}

sg:
	$(AWS_COMMAND_PREFIX) cloudformation validate-template \
		--profile ${PROFILE} \
		--template-body file://templates/sg.yml && \
	$(AWS_COMMAND_PREFIX) cloudformation deploy \
		--profile ${PROFILE} \
		--template-file ./templates/sg.yml \
		--stack-name $(PREFIX)-$(ENV)-sg \
		--region $(REGION) \
		--parameter-overrides \
		Prefix=$(PREFIX) \
		Environment=$(ENV) \
		VPCStackName=$(PREFIX)-$(ENV)-vpc

ec2:
	$(AWS_COMMAND_PREFIX) cloudformation validate-template \
		--profile ${PROFILE} \
		--template-body file://templates/ec2.yml && \
	$(AWS_COMMAND_PREFIX) cloudformation deploy \
		--profile ${PROFILE} \
		--template-file ./templates/ec2.yml \
		--stack-name $(PREFIX)-$(ENV)-ec2 \
		--region $(REGION) \
		--parameter-overrides \
		Prefix=$(PREFIX) \
		Environment=$(ENV) \
		VPCStackName=$(PREFIX)-$(ENV)-vpc \
		SGStackName=${PREFIX}-${ENV}-sg \
		AMIImageId=${AMI_IMAGE_ID} \
		Region=${REGION}

delete:
	@read -p "Enter stack name to delete [alb, ec2, rds, sg, vpc]:" ans; \
	$(AWS_COMMAND_PREFIX) cloudformation delete-stack \
		--profile ${PROFILE} \
		--stack-name ${PREFIX}-${ENV}-$$ans
