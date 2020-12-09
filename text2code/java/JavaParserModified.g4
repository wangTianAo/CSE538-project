parser grammar JavaParserModified ;
options { tokenVocab=JavaLexer ;  }
compilationUnit :  EOF|star_0 EOF|star_1 EOF|star_1 star_0 EOF|packageDeclaration EOF|packageDeclaration star_0 EOF|packageDeclaration star_1 EOF|packageDeclaration star_1 star_0 EOF;
identifierNT:   IDENTIFIER ;
packageDeclaration :  PACKAGE qualifiedName ';'|star_2 PACKAGE qualifiedName ';';
nt_0:    '.' '*' ;
importDeclaration :  IMPORT qualifiedName ';'|IMPORT qualifiedName nt_0 ';'|IMPORT STATIC qualifiedName ';'|IMPORT STATIC qualifiedName nt_0 ';';
nt_1:    classDeclaration | enumDeclaration | interfaceDeclaration | annotationTypeDeclaration ;
typeDeclaration :  nt_1|star_3 nt_1| ';' ;
modifier :   classOrInterfaceModifier | NATIVE | SYNCHRONIZED | TRANSIENT | VOLATILE ;
classOrInterfaceModifier :   annotation | PUBLIC | PROTECTED | PRIVATE | STATIC | ABSTRACT | FINAL | STRICTFP ;
variableModifier :   FINAL | annotation ;
nt_2:    EXTENDS typeType ;
nt_43:    IMPLEMENTS typeList ;
classDeclaration :  CLASS identifierNT classBody|CLASS identifierNT nt_43 classBody|CLASS identifierNT nt_2 classBody|CLASS identifierNT nt_2 nt_43 classBody|CLASS identifierNT typeParameters classBody|CLASS identifierNT typeParameters nt_43 classBody|CLASS identifierNT typeParameters nt_2 classBody|CLASS identifierNT typeParameters nt_2 nt_43 classBody;
nt_3:    ',' typeParameter ;
typeParameters : '<' typeParameter '>'|'<' typeParameter star_4 '>';
nt_4:    EXTENDS typeBound ;
typeParameter :  identifierNT|identifierNT nt_4|star_2 identifierNT|star_2 identifierNT nt_4;
nt_5:    '&' typeType ;
typeBound :  typeType|typeType star_5;
nt_6:    IMPLEMENTS typeList ;
enumDeclaration :  ENUM identifierNT '{' '}'|ENUM identifierNT '{' enumBodyDeclarations '}'|ENUM identifierNT '{' ',' '}'|ENUM identifierNT '{' ',' enumBodyDeclarations '}'|ENUM identifierNT '{' enumConstants '}'|ENUM identifierNT '{' enumConstants enumBodyDeclarations '}'|ENUM identifierNT '{' enumConstants ',' '}'|ENUM identifierNT '{' enumConstants ',' enumBodyDeclarations '}'|ENUM identifierNT nt_6 '{' '}'|ENUM identifierNT nt_6 '{' enumBodyDeclarations '}'|ENUM identifierNT nt_6 '{' ',' '}'|ENUM identifierNT nt_6 '{' ',' enumBodyDeclarations '}'|ENUM identifierNT nt_6 '{' enumConstants '}'|ENUM identifierNT nt_6 '{' enumConstants enumBodyDeclarations '}'|ENUM identifierNT nt_6 '{' enumConstants ',' '}'|ENUM identifierNT nt_6 '{' enumConstants ',' enumBodyDeclarations '}';
nt_7:    ',' enumConstant ;
enumConstants :  enumConstant|enumConstant star_6;
enumConstant :  identifierNT|identifierNT classBody|identifierNT arguments|identifierNT arguments classBody|star_2 identifierNT|star_2 identifierNT classBody|star_2 identifierNT arguments|star_2 identifierNT arguments classBody;
enumBodyDeclarations : ';'|';' star_7;
nt_8:    EXTENDS typeList ;
interfaceDeclaration :  INTERFACE identifierNT interfaceBody|INTERFACE identifierNT nt_8 interfaceBody|INTERFACE identifierNT typeParameters interfaceBody|INTERFACE identifierNT typeParameters nt_8 interfaceBody;
classBody : '{' '}'|'{' star_7 '}';
interfaceBody : '{' '}'|'{' star_8 '}';
classBodyDeclaration :   ';' |block|STATIC block|memberDeclaration|star_9 memberDeclaration;
memberDeclaration :   methodDeclaration | genericMethodDeclaration | fieldDeclaration | constructorDeclaration | genericConstructorDeclaration | interfaceDeclaration | annotationTypeDeclaration | classDeclaration | enumDeclaration ;
nt_9:    '[' ']' ;
nt_44:    THROWS qualifiedNameList ;
methodDeclaration :  typeTypeOrVoid identifierNT formalParameters methodBody|typeTypeOrVoid identifierNT formalParameters nt_44 methodBody|typeTypeOrVoid identifierNT formalParameters star_10 methodBody|typeTypeOrVoid identifierNT formalParameters star_10 nt_44 methodBody;
methodBody :   block | ';' ;
typeTypeOrVoid :   typeType | VOID ;
genericMethodDeclaration :   typeParameters methodDeclaration ;
genericConstructorDeclaration :   typeParameters constructorDeclaration ;
nt_10:    THROWS qualifiedNameList ;
constructorDeclaration :  identifierNT formalParameters constructorBody=block|identifierNT formalParameters nt_10 constructorBody=block;
fieldDeclaration :   typeType variableDeclarators ';' ;
interfaceBodyDeclaration :  interfaceMemberDeclaration|star_9 interfaceMemberDeclaration| ';' ;
interfaceMemberDeclaration :   constDeclaration | interfaceMethodDeclaration | genericInterfaceMethodDeclaration | interfaceDeclaration | annotationTypeDeclaration | classDeclaration | enumDeclaration ;
nt_11:    ',' constantDeclarator ;
constDeclaration :  typeType constantDeclarator ';'|typeType constantDeclarator star_11 ';';
nt_12:    '[' ']' ;
constantDeclarator :  identifierNT '=' variableInitializer|identifierNT star_12 '=' variableInitializer;
nt_13:    typeTypeOrVoid |typeParameters typeTypeOrVoid|typeParameters star_2 typeTypeOrVoid;
nt_45:    '[' ']' ;
nt_56:    THROWS qualifiedNameList ;
interfaceMethodDeclaration :  nt_13 identifierNT formalParameters methodBody|nt_13 identifierNT formalParameters nt_56 methodBody|nt_13 identifierNT formalParameters star_13 methodBody|nt_13 identifierNT formalParameters star_13 nt_56 methodBody|star_14 nt_13 identifierNT formalParameters methodBody|star_14 nt_13 identifierNT formalParameters nt_56 methodBody|star_14 nt_13 identifierNT formalParameters star_13 methodBody|star_14 nt_13 identifierNT formalParameters star_13 nt_56 methodBody;
interfaceMethodModifier :   annotation | PUBLIC | ABSTRACT | DEFAULT | STATIC | STRICTFP ;
genericInterfaceMethodDeclaration :   typeParameters interfaceMethodDeclaration ;
nt_14:    ',' variableDeclarator ;
variableDeclarators :  variableDeclarator|variableDeclarator star_15;
nt_15:    '=' variableInitializer ;
variableDeclarator :  variableDeclaratorId|variableDeclaratorId nt_15;
nt_16:    '[' ']' ;
variableDeclaratorId :  identifierNT|identifierNT star_16;
variableInitializer :   arrayInitializer | expression ;
nt_46:    ',' variableInitializer ;
nt_57:    ',' ;
nt_17:  variableInitializer|variableInitializer nt_57|variableInitializer star_17|variableInitializer star_17 nt_57;
arrayInitializer :  '{' '}'|'{' nt_17 '}';
nt_18:  '.' identifierNT|'.' identifierNT typeArguments;
classOrInterfaceType :  identifierNT|identifierNT star_18|identifierNT typeArguments|identifierNT typeArguments star_18;
nt_47:    EXTENDS | SUPER ;
nt_19:      nt_47   typeType ;
typeArgument :   typeType |'?'|'?' nt_19;
nt_20:    ',' qualifiedName ;
qualifiedNameList :  qualifiedName|qualifiedName star_19;
formalParameters :  '(' ')'|'(' formalParameterList ')';
nt_21:    ',' formalParameter ;
nt_48:    ',' lastFormalParameter ;
formalParameterList :  formalParameter|formalParameter nt_48|formalParameter star_20|formalParameter star_20 nt_48| lastFormalParameter ;
formalParameter :  typeType variableDeclaratorId|star_21 typeType variableDeclaratorId;
lastFormalParameter :  typeType '...' variableDeclaratorId|star_21 typeType '...' variableDeclaratorId;
nt_22:    '.' identifierNT ;
qualifiedName :  identifierNT|identifierNT star_22;
literal :   integerLiteral | floatLiteral | nt_char_literal | nt_string_literal | nt_bool_literal | nt_null_literal ;
nt_char_literal:   CHAR_LITERAL;
nt_string_literal:   STRING_LITERAL;
nt_bool_literal:   BOOL_LITERAL;
nt_null_literal:   NULL_LITERAL;
nt_decimal_literal:   DECIMAL_LITERAL;
nt_hex_literal:   HEX_LITERAL;
nt_oct_literal:   OCT_LITERAL;
nt_binary_literal:   BINARY_LITERAL;
nt_float_literal:   FLOAT_LITERAL;
nt_hex_float_literal:   HEX_FLOAT_LITERAL;
integerLiteral :   nt_decimal_literal | nt_hex_literal | nt_oct_literal | nt_binary_literal ;
floatLiteral :   nt_float_literal | nt_hex_float_literal ;
nt_49:     elementValuePairs | elementValue  ;
nt_23:  '(' ')'|'(' nt_49 ')';
annotation :  '@' qualifiedName|'@' qualifiedName nt_23;
nt_24:    ',' elementValuePair ;
elementValuePairs :  elementValuePair|elementValuePair star_23;
elementValuePair :   identifierNT '=' elementValue ;
elementValue :   expression | annotation | elementValueArrayInitializer ;
nt_50:    ',' elementValue ;
nt_25:  elementValue|elementValue star_24;
nt_51:    ',' ;
elementValueArrayInitializer :  '{' '}'|'{' nt_51 '}'|'{' nt_25 '}'|'{' nt_25 nt_51 '}';
annotationTypeDeclaration :   '@' INTERFACE identifierNT annotationTypeBody ;
nt_26:    annotationTypeElementDeclaration ;
annotationTypeBody : '{' '}'|'{' star_25 '}';
annotationTypeElementDeclaration :  annotationTypeElementRest|star_9 annotationTypeElementRest| ';' ;
annotationTypeElementRest :   typeType annotationMethodOrConstantRest ';' |classDeclaration|classDeclaration ';'|interfaceDeclaration|interfaceDeclaration ';'|enumDeclaration|enumDeclaration ';'|annotationTypeDeclaration|annotationTypeDeclaration ';';
annotationMethodOrConstantRest :   annotationMethodRest | annotationConstantRest ;
annotationMethodRest :  identifierNT '(' ')'|identifierNT '(' ')' defaultValue;
annotationConstantRest :   variableDeclarators ;
defaultValue :   DEFAULT elementValue ;
block : '{' '}'|'{' star_26 '}';
blockStatement :   localVariableDeclaration ';' | statement | localTypeDeclaration ;
localVariableDeclaration :  typeType variableDeclarators|star_21 typeType variableDeclarators;
nt_27:    classDeclaration | interfaceDeclaration ;
localTypeDeclaration :  nt_27|star_3 nt_27| ';' ;
nt_28:    ':' expression ;
nt_52:    ELSE statement ;
nt_58: star_27|star_27 finallyBlock| finallyBlock ;
statement :   blockLabel=block |ASSERT expression ';'|ASSERT expression nt_28 ';'|IF parExpression statement|IF parExpression statement nt_52| FOR '(' forControl ')' statement | WHILE parExpression statement | DO statement WHILE parExpression ';' | TRY block   nt_58  |TRY resourceSpecification block|TRY resourceSpecification block finallyBlock|TRY resourceSpecification block star_27|TRY resourceSpecification block star_27 finallyBlock|SWITCH parExpression '{' '}'|SWITCH parExpression '{' star_28 '}'|SWITCH parExpression '{' star_29 '}'|SWITCH parExpression '{' star_29 star_28 '}'| SYNCHRONIZED parExpression block |RETURN ';'|RETURN expression ';'| THROW expression ';' |BREAK ';'|BREAK identifierNT ';'|CONTINUE ';'|CONTINUE identifierNT ';'| SEMI | statementExpression=expression ';' | identifierLabel=identifierNT ':' statement ;
catchClause :  CATCH '(' catchType identifierNT ')' block|CATCH '(' star_21 catchType identifierNT ')' block;
nt_29:    '|' qualifiedName ;
catchType :  qualifiedName|qualifiedName star_30;
finallyBlock :   FINALLY block ;
resourceSpecification :  '(' resources ')'|'(' resources ';' ')';
nt_30:    ';' resource ;
resources :  resource|resource star_31;
resource :  classOrInterfaceType variableDeclaratorId '=' expression|star_21 classOrInterfaceType variableDeclaratorId '=' expression;
switchBlockStatementGroup : star_28 star_26;
nt_31:    constantExpression=expression | enumConstantName=identifierNT ;
switchLabel :   CASE   nt_31   ':' | DEFAULT ':' ;
forControl :   enhancedForControl |';' ';'|';' ';' forUpdate=expressionList|';' expression ';'|';' expression ';' forUpdate=expressionList|forInit ';' ';'|forInit ';' ';' forUpdate=expressionList|forInit ';' expression ';'|forInit ';' expression ';' forUpdate=expressionList;
forInit :   localVariableDeclaration | expressionList ;
enhancedForControl :  typeType variableDeclaratorId ':' expression|star_21 typeType variableDeclaratorId ':' expression;
parExpression :   '(' expression ')' ;
nt_32:    ',' expression ;
expressionList :  expression|expression star_32;
nt_33:    identifierNT | THIS |NEW innerCreator|NEW nonWildcardTypeArguments innerCreator| SUPER superSuffix | explicitGenericInvocation ;
nt_53:    '++' | '--' ;
nt_59:    '+'|'-'|'++'|'--' ;
nt_61:    '~'|'!' ;
nt_63:    '*'|'/'|'%' ;
nt_64:    '+'|'-' ;
nt_65:    '<' '<' | '>' '>' '>' | '>' '>' ;
nt_66:    '<=' | '>=' | '>' | '<' ;
nt_67:    '==' | '!=' ;
nt_68:    '=' | '+=' | '-=' | '*=' | '/=' | '&=' | '|=' | '^=' | '>>=' | '>>>=' | '<<=' | '%=' ;
nt_69:  identifierNT|typeArguments identifierNT| NEW ;
expression :   primary | expression '.'  nt_33  | expression '[' expression ']' |expression '(' ')'|expression '(' expressionList ')'| NEW creator | '(' typeType ')' expression | expression  nt_53  |   nt_59   expression |   nt_61   expression | expression   nt_63   expression | expression   nt_64   expression | expression   nt_65   expression | expression   nt_66   expression | expression INSTANCEOF typeType | expression   nt_67   expression | expression '&' expression | expression '^' expression | expression '|' expression | expression '&&' expression | expression '||' expression | expression '?' expression ':' expression | <assoc=right> expression  nt_68  expression | lambdaExpression  |expression '::' identifierNT|expression '::' typeArguments identifierNT| typeType '::'   nt_69  |classType '::' NEW|classType '::' typeArguments NEW;
lambdaExpression :   lambdaParameters '->' lambdaBody ;
nt_34:    ',' identifierNT ;
lambdaParameters :   identifierNT |'(' ')'|'(' formalParameterList ')'|'(' identifierNT ')'|'(' identifierNT star_33 ')';
lambdaBody :   expression | block ;
nt_35:    explicitGenericInvocationSuffix | THIS arguments ;
primary :   '(' expression ')' | THIS | SUPER | literal | identifierNT | typeTypeOrVoid '.' CLASS | nonWildcardTypeArguments   nt_35  ;
nt_36:    classOrInterfaceType '.' ;
classType :  identifierNT|identifierNT typeArguments|star_2 identifierNT|star_2 identifierNT typeArguments|nt_36 identifierNT|nt_36 identifierNT typeArguments|nt_36 star_2 identifierNT|nt_36 star_2 identifierNT typeArguments;
nt_37:    arrayCreatorRest | classCreatorRest ;
creator :   nonWildcardTypeArguments createdName classCreatorRest | createdName   nt_37  ;
nt_38:  '.' identifierNT|'.' identifierNT typeArgumentsOrDiamond;
createdName :  identifierNT|identifierNT star_34|identifierNT typeArgumentsOrDiamond|identifierNT typeArgumentsOrDiamond star_34| primitiveType ;
innerCreator :  identifierNT classCreatorRest|identifierNT nonWildcardTypeArgumentsOrDiamond classCreatorRest;
nt_54:    '[' ']' ;
nt_60:    '[' expression ']' ;
nt_62:    '[' ']' ;
nt_39:  ']' arrayInitializer|']' star_35 arrayInitializer|expression ']'|expression ']' star_36|expression ']' star_37|expression ']' star_37 star_36;
arrayCreatorRest :   '['   nt_39  ;
classCreatorRest :  arguments|arguments classBody;
explicitGenericInvocation :   nonWildcardTypeArguments explicitGenericInvocationSuffix ;
typeArgumentsOrDiamond :   '<' '>' | typeArguments ;
nonWildcardTypeArgumentsOrDiamond :   '<' '>' | nonWildcardTypeArguments ;
nonWildcardTypeArguments :   '<' typeList '>' ;
nt_40:    ',' typeType ;
typeList :  typeType|typeType star_38;
nt_41:    classOrInterfaceType | primitiveType ;
nt_55:    '[' ']' ;
typeType :  nt_41|nt_41 star_39|annotation nt_41|annotation nt_41 star_39;
primitiveType :   BOOLEAN | CHAR | BYTE | SHORT | INT | LONG | FLOAT | DOUBLE ;
nt_42:    ',' typeArgument ;
typeArguments : '<' typeArgument '>'|'<' typeArgument star_40 '>';
superSuffix :   arguments |'.' identifierNT|'.' identifierNT arguments;
explicitGenericInvocationSuffix :   SUPER superSuffix | identifierNT arguments ;
arguments :  '(' ')'|'(' expressionList ')';
star_0 : typeDeclaration | star_0 star_0;
star_1 : importDeclaration | star_1 star_1;
star_2 : annotation | star_2 star_2;
star_3 : classOrInterfaceModifier | star_3 star_3;
star_4 : nt_3 | star_4 star_4;
star_5 : nt_5 | star_5 star_5;
star_6 : nt_7 | star_6 star_6;
star_7 : classBodyDeclaration | star_7 star_7;
star_8 : interfaceBodyDeclaration | star_8 star_8;
star_9 : modifier | star_9 star_9;
star_10 : nt_9 | star_10 star_10;
star_11 : nt_11 | star_11 star_11;
star_12 : nt_12 | star_12 star_12;
star_13 : nt_45 | star_13 star_13;
star_14 : interfaceMethodModifier | star_14 star_14;
star_15 : nt_14 | star_15 star_15;
star_16 : nt_16 | star_16 star_16;
star_17 : nt_46 | star_17 star_17;
star_18 : nt_18 | star_18 star_18;
star_19 : nt_20 | star_19 star_19;
star_20 : nt_21 | star_20 star_20;
star_21 : variableModifier | star_21 star_21;
star_22 : nt_22 | star_22 star_22;
star_23 : nt_24 | star_23 star_23;
star_24 : nt_50 | star_24 star_24;
star_25 : nt_26 | star_25 star_25;
star_26 : blockStatement | star_26 star_26;
star_27 : catchClause | star_27 star_27;
star_28 : switchLabel | star_28 star_28;
star_29 : switchBlockStatementGroup | star_29 star_29;
star_30 : nt_29 | star_30 star_30;
star_31 : nt_30 | star_31 star_31;
star_32 : nt_32 | star_32 star_32;
star_33 : nt_34 | star_33 star_33;
star_34 : nt_38 | star_34 star_34;
star_35 : nt_54 | star_35 star_35;
star_36 : nt_62 | star_36 star_36;
star_37 : nt_60 | star_37 star_37;
star_38 : nt_40 | star_38 star_38;
star_39 : nt_55 | star_39 star_39;
star_40 : nt_42 | star_40 star_40;
