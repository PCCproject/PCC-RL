package unocard;

import game.SpecialEffectsManager;

public class NumberCard extends UnoCard{
	
	private final Color color;
    private final int content;
    
    public enum Color {
    	RED, 
        YELLOW,
        GREEN,
        BLUE
    }
    
    public NumberCard(Color color, int content) {
    	this.color = color;
    	this.content = content;
    }
    
    @Override
    public String getContent() {
    	return String.valueOf(this.content);
    }
    
    @Override
    public String getColor() {
    	return this.color.toString();
    }
    
    @Override
    public boolean doMatch(UnoCard ucard) {
    	if(ucard.getColor().compareTo("BLACK") == 0) {
    		return true;
    	}
    	else if(ucard.getColor().compareTo(this.getColor()) == 0) {
    		return true;
    	}
    	else if(ucard.getContent().compareTo(this.getContent()) == 0) {
    		return true;
    	}
    	else {
    		return false;
    	}
    }
    
    public void doSpecialEffect(SpecialEffectsManager effManager) {
    	return;
    }
    
	
}
